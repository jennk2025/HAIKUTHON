"""ViLT 재정렬 실험: 상위 2장만 원문-이미지 일치 점수로 다시 본다 (OWL-ViT 실험과 같은 기준·같은 후보).

- 공개 가중치 `dandelin/vilt-b32-finetuned-coco`(COCO 이미지-문장 검색으로 학습, Apache-2.0, 모델 카드 확인 필요)를
  로컬에서 실행한다. 우리 데이터로 추가 학습하지 않는다. 외부 API 없음.
- ViLT는 문장과 이미지를 함께 넣어 일치 점수 하나를 낸다. 점수가 높다고 좌우 관계를 이해했다고 단정할 수는 없다.
- 기준 = SigLIP-base-384 + 조건 재정렬(B). 상위 2장 점수에 w × ViLT 점수를 더해 두 장의 순서만 다시 정한다.
- D = ViLT로 다시 정한 상위 2장에 OWL-ViT 관계 규칙을 이어서 적용 (OWL 기준값은 OWL 실험에서 개발로 정한 값).
- w는 개발(학습 분할)에서 정하고 검증 800질의에서 한 번만 보고한다.
"""

import math
import time

import torch
from PIL import Image

from haiku.owl import relation_score, spatial_relations

VILT_ID = "dandelin/vilt-b32-finetuned-coco"


class ViltScorer:
    def __init__(self, model_id=VILT_ID, device="cuda", local_files_only=False):
        from transformers import ViltForImageAndTextRetrieval, ViltProcessor
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = ViltForImageAndTextRetrieval.from_pretrained(model_id, dtype=self.dtype,
                                                                  local_files_only=local_files_only).to(self.device).eval()
        self.proc = ViltProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.cache, self.n, self.sec = {}, 0, 0.0

    @torch.inference_mode()
    def score(self, pairs, bs=32):
        """pairs: [(image_path, text)] → 일치 점수(로짓) 목록."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        t0 = time.perf_counter()
        for s in range(0, len(todo), bs):
            chunk = todo[s:s + bs]
            ims = []
            for path, _ in chunk:
                with Image.open(path) as im:
                    ims.append(im.convert("RGB"))
            inp = self.proc(images=ims, text=[t for _, t in chunk], return_tensors="pt", padding=True,
                            truncation=True, max_length=40).to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
            logits = self.model(**inp).logits.float().view(-1).cpu().tolist()
            for p, v in zip(chunk, logits):
                self.cache[p] = v
            self.n += len(chunk)
        self.sec += time.perf_counter() - t0
        return [self.cache[p] for p in pairs]


def _baseline(rows, split, img_fn, txt_fn, lam_cond, K):
    from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores
    by_id = {r["id"]: r for r in rows}
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    ids = tr + va
    pos = {i: k for k, i in enumerate(ids)}
    img = img_fn([by_id[i]["image_path"] for i in ids])
    txt = txt_fn([by_id[i]["prompt"] for i in ids])
    conds = {i: extract_conditions(by_id[i]["prompt"]) for i in ids}
    bank = PhraseBank(txt_fn)
    bank.add([p for i in ids for p in all_phrases(conds[i])])

    def build(q_ids, g_ids):
        G = img[[pos[g] for g in g_ids]]
        items = []
        for q in q_ids:
            s = txt[pos[q]] @ G.T
            top = s.topk(min(K, len(g_ids)))
            new = top.values
            if conds[q]:
                new = rerank_scores(top.values, conds[q], condition_margins(conds[q], bank, G[top.indices]), "B", lam_cond, {})
            o = new.argsort(descending=True)
            order = [g_ids[j] for j in top.indices[o].tolist()]
            items.append({"q": q, "order": order, "s2": new[o][:2].float().cpu().tolist(),
                          "gold_rank": order.index(q) + 1 if q in order else K + 1})
        return items
    return by_id, build(tr, tr), build(va, va)


def run_vilt_experiment(rows, split, img_fn, txt_fn, vilt, owl_chk=None, owl_params=None, lam_cond=3.2, K=10,
                        w_grid=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)):
    by_id, dev, val = _baseline(rows, split, img_fn, txt_fn, lam_cond, K)
    for items in (dev, val):
        pairs = [(by_id[g]["image_path"], by_id[it["q"]]["prompt"]) for it in items for g in it["order"][:2]]
        sc = vilt.score(pairs)
        for n, it in enumerate(items):
            it["v2"] = sc[2 * n:2 * n + 2]
    score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)

    def after_vilt(items, w):
        out = []
        for it in items:
            r, top2 = it["gold_rank"], list(it["order"][:2])
            if len(top2) == 2 and it["s2"][1] + w * it["v2"][1] > it["s2"][0] + w * it["v2"][0]:
                top2 = top2[::-1]
                r = 2 if r == 1 else (1 if r == 2 else r)
            out.append((r, top2))
        return out

    def after_owl(items, prev):
        if owl_chk is None or not owl_params:
            return [r for r, _ in prev]
        out = []
        for it, (r, top2) in zip(items, prev):
            rels = spatial_relations(by_id[it["q"]]["prompt"])
            if rels:
                names = tuple(sorted({n for _, x, y in rels for n in (x, y)}))
                det = owl_chk.detect([(by_id[g]["image_path"], names) for g in top2])
                sa, sb = (relation_score(det[(by_id[g]["image_path"], names)], rels, owl_params["thr"], owl_params["margin"])
                          for g in top2)
                if sb > sa:
                    r = 2 if r == 1 else (1 if r == 2 else r)
            out.append(r)
        return out

    grid = {w: score([r for r, _ in after_vilt(dev, w)]) for w in w_grid}
    w = max(grid, key=grid.get)
    base = [it["gold_rank"] for it in val]
    v = after_vilt(val, w)
    rk_c = [r for r, _ in v]
    rk_d = after_owl(val, v)
    met = lambda rk: {"score": round(score(rk), 4), "recall@1": round(sum(r == 1 for r in rk) / len(rk), 4),
                      "fixed_2to1": sum(b == 2 and r == 1 for b, r in zip(base, rk)),
                      "broken_1to2": sum(b == 1 and r == 2 for b, r in zip(base, rk))}
    return {"chosen_on_dev": {"w": w, "dev_by_w": {str(k): round(x, 4) for k, x in grid.items()},
                              "dev_base": round(score([it["gold_rank"] for it in dev]), 4)},
            "val_A": met(base), "val_C_ViLT": met(rk_c), "val_D_ViLT+OWL": met(rk_d) if owl_chk and owl_params else None,
            "val_gold_exactly_2nd": sum(b == 2 for b in base),
            "vilt_pairs_per_sec": round(vilt.n / max(vilt.sec, 1e-6), 1),
            "examples_fixed": [val[n]["q"] for n in range(len(val)) if base[n] == 2 and rk_c[n] == 1][:10],
            "examples_broken": [val[n]["q"] for n in range(len(val)) if base[n] == 1 and rk_c[n] == 2][:10]}


def probe_vilt(vilt, rows, train_ids, pairs_csv, contrast_csv, per_cat=100):
    """2×2 점검 (학습 분할 안): 두 설명(또는 대조 문구)이 각자 자기 이미지를 더 높게 보는지. itm.probe_2x2와 같은 기준."""
    from haiku.itm import load_probe_pairs, probe_2x2
    by_id = {r["id"]: r for r in rows}
    orig, phr = load_probe_pairs(pairs_csv, contrast_csv, train_ids, per_cat)
    fn = lambda texts, ids: torch.tensor(vilt.score([(by_id[i]["image_path"], t) for t, i in zip(texts, ids)]))
    return probe_2x2(orig, fn), probe_2x2(phr, fn)
