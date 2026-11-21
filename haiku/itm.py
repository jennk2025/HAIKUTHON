"""BLIP ITM(이미지-문장 매칭) 재정렬 실험.

- 공개 가중치 `Salesforce/blip-itm-base-coco`(BSD-3-Clause, 모델 카드 확인 필요)를 로컬에서 실행한다. 외부 API 없음.
- 후보 이미지의 BLIP 비전 특징은 한 번만 계산하고, (문장, 이미지) 쌍마다 매칭 로짓 차이(match - no match)를 낸다.
- 2×2 점검: 학습 분할 안의 "속성 하나만 다른" 두 이미지와 두 원본 설명(또는 규칙 대조 문구)으로
  각 설명이 자기 이미지를 더 높게 보는지 확인한다. SigLIP과 같은 쌍에서 나란히 비교한다.
- 재정렬: A 원래 SigLIP 검색 / B + 원문 ITM / B_gate 박빙 질의에만 B / C B + 조건별 ITM.
  가중치와 박빙 기준은 학습 분할(개발)에서 정하고, 검증 분할에서 한 번만 보고한다.
- 순위는 현재 질의 문장과 그 질의의 후보 이미지로만 계산한다. 후보의 원래 설명·다른 질의 결과는 쓰지 않는다.
"""

import csv
import math
import time

import torch
from PIL import Image

from haiku.rerank import condition_weights, extract_conditions, wilson_lower

BLIP_ID = "Salesforce/blip-itm-base-coco"
ITM_KINDS = {"color_binding", "count_binding", "relation_left_right", "relation_above_below", "relation_on_inside", "state"}


class BlipITM:
    def __init__(self, model_id=BLIP_ID, device="cuda", local_files_only=False):
        from transformers import AutoProcessor, BlipForImageTextRetrieval
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = BlipForImageTextRetrieval.from_pretrained(model_id, dtype=dtype,
                                                               local_files_only=local_files_only).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.dtype = dtype

    @torch.inference_mode()
    def image_embeds(self, paths, bs=32):
        """후보 이미지 → BLIP 비전 특징 (n, 토큰, 차원). CPU 메모리에 fp16으로 보관 (미리 할당해 두 배 사용을 막음)."""
        store = None
        for s in range(0, len(paths), bs):
            ims = []
            for p in paths[s:s + bs]:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            px = self.proc.image_processor(images=ims, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            h = self.model.vision_model(pixel_values=px).last_hidden_state.half().cpu()
            if store is None:
                store = torch.empty((len(paths),) + tuple(h.shape[1:]), dtype=torch.float16)
            store[s:s + len(h)] = h
        return store

    @torch.inference_mode()
    def score_idx(self, texts, idx, feats, bs=64):
        """(문장 i, 이미지 특징 feats[idx[i]]) 쌍마다 매칭 로짓 차이. 특징은 묶음마다 꺼내 써서 메모리를 아낀다."""
        out = []
        for s in range(0, len(texts), bs):
            tok = self.proc.tokenizer(texts[s:s + bs], padding=True, truncation=True, max_length=72,
                                      return_tensors="pt").to(self.device)
            img = feats[idx[s:s + bs]].to(self.device, self.dtype)
            att = torch.ones(img.shape[:-1], dtype=torch.long, device=self.device)
            h = self.model.text_encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                        encoder_hidden_states=img, encoder_attention_mask=att).last_hidden_state
            logit = self.model.itm_head(h[:, 0, :]).float()
            out.append((logit[:, 1] - logit[:, 0]).cpu())
        return torch.cat(out) if out else torch.zeros(0)

    def score(self, texts, img_feats, bs=64):
        """(문장 i, 이미지 특징 i) 쌍마다 매칭 로짓 차이."""
        return self.score_idx(texts, list(range(len(texts))), img_feats, bs)


# ----------------------------------------------------------------------------- 2×2 점검
def probe_2x2(pairs, score_fn):
    """pairs: [(category, text_a, text_b, id_a, id_b)]. score_fn(texts, ids) → 점수.
    성공 = 설명 A가 이미지 A를, 설명 B가 이미지 B를 더 높게 평가."""
    texts, ids = [], []
    for _, ta, tb, a, b in pairs:
        texts += [ta, ta, tb, tb]
        ids += [a, b, a, b]
    s = score_fn(texts, ids).view(-1, 4)
    stats = {}
    for (cat, *_), row in zip(pairs, s.tolist()):
        aa, ab, ba, bb = row
        ok_a, ok_b = aa > ab, bb > ba
        st = stats.setdefault(cat, {"pairs": 0, "both": 0, "directions": 0, "same_image": 0})
        st["pairs"] += 1
        st["both"] += ok_a and ok_b
        st["directions"] += ok_a + ok_b
        st["same_image"] += ok_a != ok_b
    for st in stats.values():
        n = 2 * st["pairs"]
        st["both_ratio"] = f"{st['both']}/{st['pairs']}"
        st["direction_acc"] = round(st["directions"] / n, 4)
        st["direction_acc_lower95"] = round(wilson_lower(st["directions"], n), 4)
    return stats


def load_probe_pairs(pairs_csv, contrast_csv, train_ids, per_cat=300, seed=0):
    """학습 분할 안의 비교 쌍. (1) 원본 설명 2개 (단일 확정 쌍과 결합 교차 쌍), (2) 규칙 대조 문구."""
    import random
    tr = set(train_ids)
    rng = random.Random(seed)
    orig = {}
    for r in csv.DictReader(open(pairs_csv, encoding="utf-8-sig")):
        if r["id_a"] not in tr or r["id_b"] not in tr:
            continue
        cats = []
        if r["single_status"] == "단일 확정":
            cats.append("single:" + r["object_categories"])
        for part in filter(None, r["binding_swap"].split(" | ")):
            cats.append("swap:" + ("color" if part.startswith("색") else "count"))
        for c in cats:
            orig.setdefault(c, []).append((c, r["prompt_a"], r["prompt_b"], r["id_a"], r["id_b"]))
    phr = {}
    for r in csv.DictReader(open(contrast_csv, encoding="utf-8-sig")):
        if r["id_a"] in tr and r["id_b"] in tr:
            phr.setdefault(r["category"], []).append((r["category"], r["target"], r["contrast"], r["id_a"], r["id_b"]))
    pick = lambda d: [x for v in d.values() for x in (rng.sample(v, per_cat) if len(v) > per_cat else v)]
    return pick(orig), pick(phr)


# ----------------------------------------------------------------------------- 재정렬 실험
def run_itm_experiment(rows, split, pairs_csv, contrast_csv, siglip_img, siglip_txt, itm, K_list=(2, 5),
                       lam_grid=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0), lam2_grid=(0.002, 0.005, 0.01, 0.02, 0.05),
                       tau_quantiles=(0.2, 0.4, 0.6, 0.8), per_cat=300):
    """siglip_img/siglip_txt: 학습+검증 4,000개에 대한 SigLIP 정규화 임베딩 (split의 train_ids + valid_ids 순서)."""
    from haiku.metrics import metrics_from_ranks
    by_id = {r["id"]: r for r in rows}
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    ids = tr + va
    pos = {i: k for k, i in enumerate(ids)}
    out = {}

    # BLIP 비전 특징: 학습+검증 이미지 (시간 측정)
    t0 = time.perf_counter()
    feats = itm.image_embeds([by_id[i]["image_path"] for i in ids])
    out["blip_image_sec"] = round(time.perf_counter() - t0, 1)
    out["blip_img_per_sec"] = round(len(ids) / out["blip_image_sec"], 1)

    def itm_fn(texts, img_ids):
        return itm.score_idx(texts, [pos[i] for i in img_ids], feats)

    def siglip_fn(texts, img_ids):
        T = siglip_txt_phr(texts)
        I = siglip_img[[pos[i] for i in img_ids]].to(T)
        return (T * I).sum(-1).cpu()

    cache = {}

    def siglip_txt_phr(texts):
        new = [t for t in dict.fromkeys(texts) if t not in cache]
        if new:
            for t, v in zip(new, siglip_txt_encode(new)):
                cache[t] = v
        return torch.stack([cache[t] for t in texts])

    siglip_txt_encode = siglip_txt
    # 2×2 점검 (학습 분할)
    orig_pairs, phr_pairs = load_probe_pairs(pairs_csv, contrast_csv, tr, per_cat)
    t0 = time.perf_counter()
    out["probe_original_BLIP"] = probe_2x2(orig_pairs, itm_fn)
    out["itm_pairs_per_sec"] = round(4 * len(orig_pairs) / (time.perf_counter() - t0), 1)
    out["probe_original_SigLIP"] = probe_2x2(orig_pairs, siglip_fn)
    out["probe_phrase_BLIP"] = probe_2x2(phr_pairs, itm_fn)
    out["probe_phrase_SigLIP"] = probe_2x2(phr_pairs, siglip_fn)
    from haiku.rerank import PROBE_TO_KIND
    reliability = {}
    for cat, st in out["probe_phrase_BLIP"].items():
        kind = PROBE_TO_KIND.get(cat)
        if kind:
            reliability[kind] = round(max(0.0, (st["direction_acc_lower95"] - 0.5) / 0.5), 4)
    out["itm_reliability"] = reliability

    # 질의 원문 텍스트 임베딩 (SigLIP)
    qtxt = siglip_txt_phr([by_id[i]["prompt"] for i in ids])
    conds = {i: [c for c in extract_conditions(by_id[i]["prompt"]) if c["kind"] in ITM_KINDS] for i in ids}

    def prepare(q_ids, g_ids, K):
        G = siglip_img[[pos[g] for g in g_ids]]
        gpos = {g: k for k, g in enumerate(g_ids)}
        items, texts, img_ids = [], [], []
        for q in q_ids:
            s = (qtxt[pos[q]].to(G) @ G.T).cpu()
            top = s.topk(K)
            cand = [g_ids[j] for j in top.indices.tolist()]
            items.append({"q": q, "cand": cand, "s": top.values, "gold": q, "base_rank": int((s >= s[gpos[q]]).sum()),
                          "gap": float(top.values[0] - top.values[1])})
            texts += [by_id[q]["prompt"]] * K
            img_ids += cand
        t0 = time.perf_counter()
        itm_orig = itm_fn(texts, img_ids).view(len(q_ids), K)
        sec_orig = time.perf_counter() - t0
        # 조건별 ITM 점수 (목표 - 대조)
        ctexts, cimg, owners = [], [], []
        for n, it in enumerate(items):
            for c_i, c in enumerate(conds[it["q"]]):
                for j, g in enumerate(it["cand"]):
                    ctexts += [c["target"], c["contrasts"][0]]
                    cimg += [g, g]
                    owners.append((n, c_i, j))
        t0 = time.perf_counter()
        cs = itm_fn(ctexts, cimg).view(-1, 2) if ctexts else torch.zeros(0, 2)
        sec_cond = time.perf_counter() - t0
        for n, it in enumerate(items):
            it["itm"] = itm_orig[n]
            it["M"] = torch.zeros(len(conds[it["q"]]), K)
        for (n, c_i, j), v in zip(owners, (cs[:, 0] - cs[:, 1]).tolist()):
            items[n]["M"][c_i, j] = v
        return items, {"itm_orig_sec": round(sec_orig, 2), "itm_cond_sec": round(sec_cond, 2),
                       "itm_orig_pairs": len(texts), "itm_cond_pairs": len(ctexts)}

    def ranks(items, method, lam=0.0, lam2=0.0, tau=None):
        rk, top1 = [], []
        for it in items:
            new = it["s"].clone()
            apply = method != "A" and (tau is None or it["gap"] < tau)
            if apply:
                new = new + lam * it["itm"]
                if method == "C" and len(it["M"]):
                    w = condition_weights(conds[it["q"]], it["M"], "C", reliability)
                    if w is not None:
                        new = new + lam2 * (w @ it["M"])
            order = [it["cand"][j] for j in new.argsort(descending=True).tolist()]
            rk.append(order.index(it["gold"]) + 1 if it["gold"] in order else it["base_rank"])
            top1.append(order[0])
        return rk, top1

    score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)
    out["by_K"] = {}
    for K in K_list:
        dev, dev_t = prepare(tr, tr, K)
        val, val_t = prepare(va, va, K)
        res = {"dev_timing": dev_t, "val_timing": val_t,
               "dev_hit@K": round(sum(it["base_rank"] <= K for it in dev) / len(dev), 4),
               "val_hit@K": round(sum(it["base_rank"] <= K for it in val) / len(val), 4)}
        sc = {l: score(ranks(dev, "B", l)[0]) for l in lam_grid}
        lam = max(sc, key=sc.get)
        res["dev_B_by_lam"] = {str(k): round(v, 4) for k, v in sc.items()}
        gaps = sorted(it["gap"] for it in dev)
        taus = sorted({gaps[min(len(gaps) - 1, int(q * len(gaps)))] for q in tau_quantiles})
        sc_t = {t: score(ranks(dev, "B", lam, tau=t)[0]) for t in taus}
        tau = max(sc_t, key=sc_t.get)
        sc2 = {l2: score(ranks(dev, "C", lam, l2)[0]) for l2 in lam2_grid}
        lam2 = max(sc2, key=sc2.get)
        res["dev_C_by_lam2"] = {str(k): round(v, 4) for k, v in sc2.items()}
        res["chosen"] = {"lam": lam, "tau_gate": round(tau, 5), "lam2": lam2}
        base_rk, _ = ranks(val, "A")
        for name, kw in (("A", {}), ("B", {"lam": lam}), ("B_gate", {"lam": lam, "tau": tau}),
                         ("C", {"lam": lam, "lam2": lam2})):
            m = "A" if name == "A" else ("C" if name == "C" else "B")
            rk, _ = ranks(val, m, **kw)
            met = metrics_from_ranks(rk)
            res[f"val_{name}"] = {"score": round(met["score"], 4), "recall@1": round(met["recall@1"], 4),
                                  "wrong_to_right@1": sum(b > 1 and r == 1 for b, r in zip(base_rk, rk)),
                                  "right_to_wrong@1": sum(b == 1 and r > 1 for b, r in zip(base_rk, rk))}
        res["val_gate_ratio"] = round(sum(it["gap"] < tau for it in val) / len(val), 4)
        out["by_K"][str(K)] = res
    return out
