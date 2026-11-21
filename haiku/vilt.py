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


def run_vs_v3(rows, split, img_fn, txt_fn, vilt, lam_cond=3.2, K=10, k_list=(3, 5),
              lam_grid=(0.01, 0.02, 0.05, 0.1, 0.2), w_grid=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0),
              seed=0, dev_frac=0.15, epochs=3, device="cuda", limit=None):
    """v3와 같은 구조(SigLIP → 조건 재정렬 → 학습한 BLIP)를 공정하게 재현하고, 그 위에 ViLT를 더하면 좋아지는지 본다.
    - BLIP은 ITM 학습 그룹으로만 학습한다 (개발·검증 그룹은 보지 않음, [3/3]·v3 ①과 같은 분할).
    - v3의 k·λ와 ViLT 비중 w는 개발 그룹에서 정하고, 검증 800질의에서 한 번만 보고한다 (w=0이면 ViLT를 안 쓰는 것과 같음).
    - ViLT는 v3 순서의 상위 2장만 다시 본다."""
    import argparse
    import random

    from haiku.combo import prepare
    from haiku.itm import BLIP_ID
    from haiku.itm_finetune import train_itm, wrap
    from haiku.rerank import PhraseBank, all_phrases, extract_conditions
    from haiku.siblings import build_groups, parse_prompt
    dev_t = torch.device(device)
    by_id = {r["id"]: r for r in rows}
    group_of, _ = build_groups(rows, {r["id"]: parse_prompt(r["prompt"]) for r in rows})
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    groups = sorted({group_of[i] for i in tr})
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[:max(1, int(len(groups) * dev_frac))])
    dev = [i for i in tr if group_of[i] in dev_groups]
    ft = [i for i in tr if group_of[i] not in dev_groups]
    if limit:
        ft, dev, va = ft[:limit[0]], dev[:limit[1]], va[:limit[2]]
    ns = argparse.Namespace(blip=BLIP_ID, seed=seed, lr=2e-5, epochs=epochs, queries_per_step=8, negatives=7)
    model, proc, info = train_itm(rows, ft, ns, dev_t)
    itm = wrap(model, proc, dev_t)
    ids = dev + va
    pos = {i: k for k, i in enumerate(ids)}
    prompts = {i: by_id[i]["prompt"] for i in ids}
    img = img_fn([by_id[i]["image_path"] for i in ids])
    txt = txt_fn([prompts[i] for i in ids])
    conds = {i: extract_conditions(prompts[i]) for i in ids}
    bank = PhraseBank(txt_fn)
    bank.add([p for i in ids for p in all_phrases(conds[i])])
    feats = itm.image_embeds([by_id[i]["image_path"] for i in ids])
    k_max = max(k_list)
    dev_items, _ = prepare(dev, dev, txt, img, pos, conds, bank, lam_cond, K, itm, feats, prompts, k_max)
    val_items, _ = prepare(va, va, txt, img, pos, conds, bank, lam_cond, K, itm, feats, prompts, k_max)
    score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)
    rank = lambda order, it: order.index(it["gold"]) + 1 if it["gold"] in order else it["base_rank"]

    def v3_order(it, k, lam):
        head = it["sB"][:k] + lam * it["itm"][:k]
        o = head.argsort(descending=True).tolist()
        return [it["idxB"][j] for j in o] + it["idxB"][k:], head[o].tolist()

    grid = {(k, lam): score([rank(v3_order(it, k, lam)[0], it) for it in dev_items]) for k in k_list for lam in lam_grid}
    k, lam = max(grid, key=grid.get)

    def vilt_scores(items, g_ids):
        orders = [v3_order(it, k, lam) for it in items]
        pairs = [(by_id[g_ids[o[j]]]["image_path"], prompts[it["q"]]) for it, (o, _) in zip(items, orders) for j in (0, 1)]
        return orders, vilt.score(pairs)

    def with_w(items, orders, v, w):
        rk = []
        for n, (it, (order, sc)) in enumerate(zip(items, orders)):
            if sc[1] + w * v[2 * n + 1] > sc[0] + w * v[2 * n]:
                order = [order[1], order[0]] + order[2:]
            rk.append(rank(order, it))
        return rk

    d_o, d_v = vilt_scores(dev_items, dev)
    v_o, v_v = vilt_scores(val_items, va)
    wgrid = {w: score(with_w(dev_items, d_o, d_v, w)) for w in w_grid}
    w = max(wgrid, key=wgrid.get)
    search = [it["base_rank"] for it in val_items]
    v3 = with_w(val_items, v_o, v_v, 0.0)
    v3v = with_w(val_items, v_o, v_v, w)
    met = lambda rk, ref: {"score": round(score(rk), 4), "recall@1": round(sum(r == 1 for r in rk) / len(rk), 4),
                           "fixed_vs_v3": sum(b > 1 and r == 1 for b, r in zip(ref, rk)),
                           "broken_vs_v3": sum(b == 1 and r > 1 for b, r in zip(ref, rk))}
    return {"sizes": {"itm_train": len(ft), "dev": len(dev), "val": len(va)}, "train": info,
            "chosen_on_dev": {"k": k, "lam_itm": lam, "w_vilt": w, "dev_v3": round(grid[(k, lam)], 4),
                              "dev_v3+vilt": round(wgrid[w], 4)},
            "val_search_only": met(search, v3), "val_v3": met(v3, v3), "val_v3+ViLT": met(v3v, v3),
            "val_v3_gold_exactly_2nd": sum(r == 2 for r in v3)}
