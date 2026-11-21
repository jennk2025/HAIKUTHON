"""OWL-ViT 공간 관계 점검: 질의의 두 물체를 후보 이미지에서 검출해 좌우·상하 관계가 맞는지 상자 좌표로 확인한다.

- 공개 가중치 `google/owlvit-base-patch32`(Apache-2.0, 모델 카드 확인 필요)를 로컬에서 실행한다. 학습 없음, 외부 API 없음.
- 질의 문장 하나에서 규칙으로 관계(주체·방향·대상)를 뽑는다. 같은 종류 물체가 여러 개라 대상을 특정할 수 없으면 쓰지 않는다.
- 후보 이미지마다 두 물체의 가장 그럴듯한 상자를 찾고, 검출 점수가 기준보다 낮거나 두 중심이 너무 가까우면 '판단 보류'(0)로 둔다.
- 규칙: 상위 2장 중 2위 후보의 관계 점수가 1위 후보보다 높을 때만 두 장의 순서를 바꾼다 (3위 이하는 그대로).
- 검출 기준과 여유값은 개발(학습 분할)에서 정하고, 검증 800질의에서는 한 번만 보고한다.
- 순위는 현재 질의 문장과 그 질의의 후보 이미지로만 계산한다. 후보의 원래 설명·다른 질의 결과는 쓰지 않는다.
"""

import math
import time

import torch
from PIL import Image

from haiku.siblings import parse_prompt

OWL_ID = "google/owlvit-base-patch32"


def spatial_relations(prompt):
    """질의 하나 → [(rel, x, y)] (rel ∈ left_of, above). 대상이 모호한 관계는 제외."""
    p = parse_prompt(prompt)
    multi = {f.split(":", 1)[1] for f in p["flags"] if f.startswith("multi_instance:")}
    one = {n for n, o in p["objects"].items() if o["count"] == 1}              # 물체가 여러 개면 어느 것인지 애매하므로 제외
    return [(r, x, y) for r, x, y in p["relations"]
            if r in ("left_of", "above") and x not in multi and y not in multi and x in one and y in one]


class OwlChecker:
    def __init__(self, model_id=OWL_ID, device="cuda", local_files_only=False):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = OwlViTForObjectDetection.from_pretrained(model_id, dtype=dtype, local_files_only=local_files_only).to(self.device).eval()
        self.proc = OwlViTProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.dtype = dtype
        self.cache, self.n_forward, self.sec = {}, 0, 0.0

    @torch.inference_mode()
    def detect(self, items, bs=16):
        """items: [(image_path, (name, ...))] → {(path, names): {name: (score, cx, cy, w, h)}} (0~1, 아래로 갈수록 cy 증가)."""
        todo = [it for it in dict.fromkeys(items) if it not in self.cache]
        t0 = time.perf_counter()
        for s in range(0, len(todo), bs):
            chunk = todo[s:s + bs]
            ims = []
            for path, _ in chunk:
                with Image.open(path) as im:
                    ims.append(im.convert("RGB"))
            texts = [[f"a photo of a {n}" for n in names] for _, names in chunk]
            width = max(len(t) for t in texts)
            texts = [t + [" "] * (width - len(t)) for t in texts]          # 질의 수를 맞춤 (빈 칸은 쓰지 않음)
            inp = self.proc(text=texts, images=ims, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
            out = self.model(**inp)
            probs = out.logits.float().sigmoid()                           # (B, 상자 수, 질의 수)
            boxes = out.pred_boxes.float()                                 # (B, 상자 수, 4) cx, cy, w, h
            for b, (path, names) in enumerate(chunk):
                det = {}
                for q, name in enumerate(names):
                    j = int(probs[b, :, q].argmax())
                    det[name] = (float(probs[b, j, q]), float(boxes[b, j, 0]), float(boxes[b, j, 1]),
                                 float(boxes[b, j, 2]), float(boxes[b, j, 3]))
                self.cache[(path, names)] = det
            self.n_forward += len(chunk)
        self.sec += time.perf_counter() - t0
        return {it: self.cache[it] for it in items}


def relation_score(det, rels, thr, margin):
    """관계마다 +1(맞음) / -1(어김) / 0(판단 보류)의 합."""
    total = 0
    for rel, x, y in rels:
        if x not in det or y not in det:
            continue
        (sx, xx, xy), (sy, yx, yy) = det[x][:3], det[y][:3]
        if min(sx, sy) < thr:
            continue
        d = (yx - xx) if rel == "left_of" else (yy - xy)                 # x가 y의 왼쪽/위면 양수
        if d > margin:
            total += 1
        elif d < -margin:
            total -= 1
    return total


def run_owl_experiment(rows, split, siglip_img_fn, siglip_txt_fn, owl, lam_cond=3.2, K=10,
                       thr_grid=(0.1, 0.15, 0.2, 0.3, 0.4), margin_grid=(0.03, 0.06, 0.1)):
    """기준 = SigLIP-base-384 + 조건 재정렬(B). 상위 2장만 OWL 규칙으로 바꿔 보고, 개발에서 기준값을 정해 검증에서 보고한다."""
    from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores
    by_id = {r["id"]: r for r in rows}
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    ids = tr + va
    pos = {i: k for k, i in enumerate(ids)}
    img = siglip_img_fn([by_id[i]["image_path"] for i in ids])
    txt = siglip_txt_fn([by_id[i]["prompt"] for i in ids])
    conds = {i: extract_conditions(by_id[i]["prompt"]) for i in ids}
    bank = PhraseBank(siglip_txt_fn)
    bank.add([p for i in ids for p in all_phrases(conds[i])])
    rels = {i: spatial_relations(by_id[i]["prompt"]) for i in ids}

    def base(q_ids, g_ids):
        G = img[[pos[g] for g in g_ids]]
        gpos = {g: k for k, g in enumerate(g_ids)}
        items = []
        for q in q_ids:
            s = txt[pos[q]] @ G.T
            top = s.topk(min(K, len(g_ids)))
            new = top.values
            if conds[q]:
                new = rerank_scores(top.values, conds[q], condition_margins(conds[q], bank, G[top.indices]), "B", lam_cond, {})
            order = [g_ids[j] for j in top.indices[new.argsort(descending=True)].tolist()]
            gold_rank = order.index(q) + 1 if q in order else K + 1
            items.append({"q": q, "order": order, "gold_rank": gold_rank})
        return items

    dev, val = base(tr, tr), base(va, va)
    for items in (dev, val):
        need = []
        for it in items:
            if rels[it["q"]]:
                names = tuple(sorted({n for _, x, y in rels[it["q"]] for n in (x, y)}))
                it["names"] = names
                need += [(by_id[g]["image_path"], names) for g in it["order"][:2]]
        owl.detect(need)

    def apply(items, thr, margin):
        out = []
        for it in items:
            r = it["gold_rank"]
            swapped = False
            if rels[it["q"]] and len(it["order"]) >= 2:
                a, b = it["order"][:2]
                sa = relation_score(owl.cache[(by_id[a]["image_path"], it["names"])], rels[it["q"]], thr, margin)
                sb = relation_score(owl.cache[(by_id[b]["image_path"], it["names"])], rels[it["q"]], thr, margin)
                if sb > sa:
                    swapped = True
                    r = 2 if r == 1 else (1 if r == 2 else r)
            out.append((r, swapped))
        return out

    score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)
    grid = {}
    for thr in thr_grid:
        for mg in margin_grid:
            grid[(thr, mg)] = score([r for r, _ in apply(dev, thr, mg)])
    thr, mg = max(grid, key=grid.get)
    res = apply(val, thr, mg)
    base_rk = [it["gold_rank"] for it in val]
    new_rk = [r for r, _ in res]
    rel_q = [n for n, it in enumerate(val) if rels[it["q"]]]
    applied = [n for n in rel_q if res[n][1]]
    met = lambda rk: {"score": round(score(rk), 4), "recall@1": round(sum(r == 1 for r in rk) / len(rk), 4)}
    return {
        "chosen_on_dev": {"thr": thr, "margin": mg, "dev_score": round(grid[(thr, mg)], 4),
                          "dev_base": round(score([it["gold_rank"] for it in dev]), 4)},
        "val_A": met(base_rk), "val_A+OWL": met(new_rk),
        "val_fixed_2to1": sum(b == 2 and r == 1 for b, r in zip(base_rk, new_rk)),
        "val_broken_1to2": sum(b == 1 and r == 2 for b, r in zip(base_rk, new_rk)),
        "val_gold_exactly_2nd": sum(b == 2 for b in base_rk),
        "val_relation_queries": len(rel_q),
        "val_relation_gold_2nd": sum(base_rk[n] == 2 for n in rel_q),
        "val_swapped": len(applied),
        "owl_images_per_sec": round(owl.n_forward / max(owl.sec, 1e-6), 1),
        "owl_forwards": owl.n_forward,
        "examples_fixed": [val[n]["q"] for n in range(len(val)) if base_rk[n] == 2 and new_rk[n] == 1][:10],
        "examples_broken": [val[n]["q"] for n in range(len(val)) if base_rk[n] == 1 and new_rk[n] == 2][:10],
    }
