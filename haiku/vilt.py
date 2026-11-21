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
    def __init__(self, model_id=VILT_ID, device="cuda", local_files_only=False, model=None):
        """model: 이미 학습한 ViLT를 넣으면 그 모델로 채점한다 (없으면 공개 가중치)."""
        from transformers import ViltForImageAndTextRetrieval, ViltProcessor
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if model is None:
            model = ViltForImageAndTextRetrieval.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = model.to(self.device, dtype=self.dtype).eval()
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



class _ViltImages(torch.utils.data.Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.ip = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.ip(images=im, return_tensors="pt")["pixel_values"][0].half()


def train_vilt(rows, train_ids, device="cuda", epochs=3, lr=2e-5, queries_per_step=4, negatives=7, seed=0,
               model_id=VILT_ID, workers=2):
    """우리 데이터로 ViLT 학습 (BLIP 학습과 같은 방식): 질의마다 정답 1장 + 오답 7장(형제 이미지 먼저),
    8장 중 정답을 고르도록 학습(CE). train_ids의 이미지·문장만 쓴다. 모델 전체를 학습한다."""
    import random

    from transformers import ViltForImageAndTextRetrieval, ViltProcessor

    from haiku.itm_finetune import negatives_for
    from haiku.siblings import build_groups, parse_prompt
    dev_t = torch.device(device)
    by_id = {r["id"]: r for r in rows}
    group_of, members = build_groups(rows, {r["id"]: parse_prompt(r["prompt"]) for r in rows})
    proc = ViltProcessor.from_pretrained(model_id)
    torch.manual_seed(seed)
    model = ViltForImageAndTextRetrieval.from_pretrained(model_id).to(dev_t)
    t0 = time.perf_counter()
    dl = torch.utils.data.DataLoader(_ViltImages([by_id[i]["image_path"] for i in train_ids], proc.image_processor),
                                     batch_size=64, num_workers=workers)
    px = torch.cat(list(dl))                                               # (N, 3, 384, 384) fp16, CPU
    pos = {i: k for k, i in enumerate(train_ids)}
    prep_sec = time.perf_counter() - t0
    print(f"ViLT 이미지 전처리 {len(train_ids)}장 {prep_sec:.0f}초", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = dev_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pool, pool_list = set(train_ids), sorted(train_ids)
    log, t0 = [], time.perf_counter()
    for ep in range(epochs):
        rng = random.Random(seed * 1000 + ep)
        order = list(train_ids)
        rng.shuffle(order)
        model.train()
        losses = []
        for s in range(0, len(order), queries_per_step):
            qs = order[s:s + queries_per_step]
            texts, idx = [], []
            for q in qs:
                negs = negatives_for(q, group_of, members, pool, pool_list, negatives, rng)
                texts += [by_id[q]["prompt"]] * (1 + len(negs))
                idx += [pos[q]] + [pos[x] for x in negs]
            tok = proc.tokenizer(texts, padding="max_length", truncation=True, max_length=40, return_tensors="pt").to(dev_t)
            pv = px[idx].to(dev_t, non_blocking=True).float()
            mask = torch.ones(pv.shape[0], pv.shape[2], pv.shape[3], dtype=torch.long, device=dev_t)
            with torch.autocast(device_type=dev_t.type, dtype=torch.float16, enabled=use_amp):
                logit = model(**tok, pixel_values=pv, pixel_mask=mask).logits.float()
            loss = torch.nn.functional.cross_entropy(logit.view(len(qs), -1),
                                                     torch.zeros(len(qs), dtype=torch.long, device=dev_t))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        log.append({"epoch": ep + 1, "loss": round(sum(losses) / len(losses), 4)})
        print(f"  ViLT epoch {ep + 1}/{epochs} loss {log[-1]['loss']:.4f}", flush=True)
    model.eval()
    del opt, px
    return model, {"prep_sec": round(prep_sec, 1), "train_sec": round(time.perf_counter() - t0, 1), "epochs": log,
                   "train_queries": len(train_ids)}


def color_chk_conds(prompt):
    from haiku.colorcrop import crop_conditions
    return crop_conditions(prompt)


def v3_groups(rows, split, seed=0, dev_frac=0.15):
    """v3 조합 평가와 같은 분할: 학습 분할을 ITM 학습 그룹(ft)과 개발 그룹(dev)으로 나눈다. 검증(va)은 그대로."""
    import random

    from haiku.siblings import build_groups, parse_prompt
    group_of, _ = build_groups(rows, {r["id"]: parse_prompt(r["prompt"]) for r in rows})
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    groups = sorted({group_of[i] for i in tr})
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[:max(1, int(len(groups) * dev_frac))])
    return [i for i in tr if group_of[i] not in dev_groups], [i for i in tr if group_of[i] in dev_groups], va


def run_vs_v3(rows, split, img_fn, txt_fn, vilt=None, lam_cond=3.2, K=10, k_list=(3, 5),
              lam_grid=(0.01, 0.02, 0.05, 0.1, 0.2), w_grid=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
              seed=0, dev_frac=0.15, epochs=3, device="cuda", limit=None,
              owl_chk=None, thr_grid=(0.05, 0.1, 0.15, 0.2, 0.3), margin_grid=(0.03, 0.06, 0.1),
              color_chk=None, mu_grid=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)):
    """v3와 같은 구조(SigLIP → 조건 재정렬 → 학습한 BLIP)를 공정하게 재현하고, 그 위에 다른 점검을 더하면 좋아지는지 본다.
    - BLIP은 ITM 학습 그룹으로만 학습한다 (개발·검증 그룹은 보지 않음, v3 ①과 같은 분할).
    - vilt: 점수기 하나 또는 {이름: 점수기}. v3 순서의 상위 2장 점수에 w × 점수를 더해 두 장의 순서만 다시 정한다.
    - owl_chk: 있으면 v3 상위 2장에 OWL 좌우·상하 규칙도 적용한다 (v4와 같은 위치).
    - color_chk: 있으면 v3 상위 2장에서 물체 영역별 색 비교(haiku.colorcrop)도 적용한다 (두 후보 모두 찾은 조건만).
    - k·λ·w·OWL 기준값·색 비중 μ는 모두 개발 그룹에서 정하고, 검증 800질의에서 한 번만 보고한다."""
    import argparse

    from haiku.combo import prepare
    from haiku.itm import BLIP_ID
    from haiku.itm_finetune import train_itm, wrap
    from haiku.rerank import PhraseBank, all_phrases, extract_conditions
    scorers = vilt if isinstance(vilt, dict) else ({"ViLT": vilt} if vilt is not None else {})
    dev_t = torch.device(device)
    by_id = {r["id"]: r for r in rows}
    ft, dev, va = v3_groups(rows, split, seed, dev_frac)
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
    del model, itm, feats
    score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)
    rank = lambda order, it: order.index(it["gold"]) + 1 if it["gold"] in order else it["base_rank"]

    def v3_order(it, k, lam):
        head = it["sB"][:k] + lam * it["itm"][:k]
        o = head.argsort(descending=True).tolist()
        return [it["idxB"][j] for j in o] + it["idxB"][k:], head[o].tolist()

    grid = {(k, lam): score([rank(v3_order(it, k, lam)[0], it) for it in dev_items]) for k in k_list for lam in lam_grid}
    k, lam = max(grid, key=grid.get)
    d_o = [v3_order(it, k, lam) for it in dev_items]
    v_o = [v3_order(it, k, lam) for it in val_items]
    top2 = lambda items, g_ids, orders: [(by_id[g_ids[o[j]]]["image_path"], prompts[it["q"]])
                                         for it, (o, _) in zip(items, orders) for j in (0, 1)]

    def with_w(items, orders, v, w):
        rk = []
        for n, (it, (order, sc)) in enumerate(zip(items, orders)):
            if sc[1] + w * v[2 * n + 1] > sc[0] + w * v[2 * n]:
                order = [order[1], order[0]] + order[2:]
            rk.append(rank(order, it))
        return rk

    search = [it["base_rank"] for it in val_items]
    v3 = with_w(val_items, v_o, [0.0] * (2 * len(val_items)), 0.0)
    met = lambda rk: {"score": round(score(rk), 4), "recall@1": round(sum(r == 1 for r in rk) / len(rk), 4),
                      "fixed_vs_v3": sum(b > 1 and r == 1 for b, r in zip(v3, rk)),
                      "broken_vs_v3": sum(b == 1 and r > 1 for b, r in zip(v3, rk))}
    chosen = {"k": k, "lam_itm": lam, "dev_v3": round(grid[(k, lam)], 4)}
    res = {"sizes": {"itm_train": len(ft), "dev": len(dev), "val": len(va)}, "train": info,
           "val_search_only": met(search), "val_v3": met(v3), "val_v3_gold_exactly_2nd": sum(r == 2 for r in v3)}
    per_q = {"q": [it["q"] for it in val_items], "ranks": {"search": search, "v3": v3}}
    _met = met

    def met(rk, name=None):                         # 보고용 지표 + 질의별 순위 보관
        if name:
            per_q["ranks"][name] = list(rk)
        return _met(rk)
    for name, sc in scorers.items():
        t0, n0 = time.perf_counter(), getattr(sc, "n", 0)
        d_v, v_v = sc.score(top2(dev_items, dev, d_o)), sc.score(top2(val_items, va, v_o))
        wg = {w: score(with_w(dev_items, d_o, d_v, w)) for w in w_grid}
        w = max(wg, key=wg.get)
        res[f"val_v3+{name}"] = met(with_w(val_items, v_o, v_v, w), f"v3+{name}")
        chosen[f"w[{name}]"] = w
        chosen[f"dev_v3+{name}"] = round(wg[w], 4)
        res[f"pairs_per_sec[{name}]"] = round((getattr(sc, "n", 0) - n0) / max(time.perf_counter() - t0, 1e-6), 1)
    if owl_chk is not None:
        from haiku.owl import relation_score, spatial_relations

        def owl_prep(items, g_ids, orders, kinds=("left_of", "above")):
            out, need = [], []
            for it, (o, _) in zip(items, orders):
                rels = spatial_relations(prompts[it["q"]], kinds)
                if rels and len(o) >= 2:
                    names = tuple(sorted({x for _, a, b in rels for x in (a, b)}))
                    paths = [by_id[g_ids[j]]["image_path"] for j in o[:2]]
                    need += [(p, names) for p in paths]
                    out.append((rels, names, paths))
                else:
                    out.append(None)
            owl_chk.detect(need)
            return out

        def owl_ranks(items, orders, prep, thr, mg):
            rk = []
            for it, (o, _), pr in zip(items, orders, prep):
                if pr:
                    rels, names, paths = pr
                    sa, sb = (relation_score(owl_chk.cache[(p, names)], rels, thr, mg) for p in paths)
                    if sb > sa:
                        o = [o[1], o[0]] + o[2:]
                rk.append(rank(o, it))
            return rk

        t_owl0, n_owl0 = owl_chk.sec, owl_chk.n_forward
        d_p, v_p = owl_prep(dev_items, dev, d_o), owl_prep(val_items, va, v_o)
        og = {(t, m): score(owl_ranks(dev_items, d_o, d_p, t, m)) for t in thr_grid for m in margin_grid}
        t, m = max(og, key=og.get)
        res["val_v3+OWL"] = met(owl_ranks(val_items, v_o, v_p, t, m), "v3+OWL")
        res["val_v3+OWL(v4값 0.1/0.06)"] = met(owl_ranks(val_items, v_o, v_p, 0.1, 0.06), "v3+OWL(v4값 0.1/0.06)")
        chosen.update({"owl_thr": t, "owl_margin": m, "dev_v3+OWL": round(og[(t, m)], 4)})
        res["val_relation_queries"] = sum(p is not None for p in v_p)
        res["owl_relation_images"] = owl_chk.n_forward - n_owl0
        res["owl_sec_per_image"] = round((owl_chk.sec - t_owl0) / max(owl_chk.n_forward - n_owl0, 1), 4)
        # 확장: 위에 놓임·안·타고 있음·옆까지 상자 규칙으로 확인 (기준값은 개발에서 따로 정함)
        from haiku.owl import EXT_RELS
        n_ext0 = owl_chk.n_forward
        d_px, v_px = owl_prep(dev_items, dev, d_o, EXT_RELS), owl_prep(val_items, va, v_o, EXT_RELS)
        ogx = {(tt, mm): score(owl_ranks(dev_items, d_o, d_px, tt, mm)) for tt in thr_grid for mm in margin_grid}
        tx, mx = max(ogx, key=ogx.get)
        res["val_v3+OWL확장"] = met(owl_ranks(val_items, v_o, v_px, tx, mx), "v3+OWL확장")
        chosen.update({"owl_ext_thr": tx, "owl_ext_margin": mx, "dev_v3+OWL확장": round(ogx[(tx, mx)], 4)})
        res["val_relation_queries_ext"] = sum(p is not None for p in v_px)
        res["owl_ext_new_images"] = owl_chk.n_forward - n_ext0
    if color_chk is not None:
        from haiku.colorcrop import pair_scores

        def color_vals(items, g_ids, orders):
            its = [(by_id[g_ids[o[j]]]["image_path"], prompts[it["q"]]) for it, (o, _) in zip(items, orders) for j in (0, 1)]
            per = color_chk.margins(its)
            v, used = [], 0
            for n in range(len(items)):
                sa, sb, u = pair_scores(per[2 * n], per[2 * n + 1], "crop")
                v += [sa, sb]
                used += u > 0
            return v, used, its

        o0, n0, c0, s0 = color_chk.owl.sec, color_chk.owl.n_forward, color_chk.n_crops, color_chk.crop_sec
        (d_c, _, _), (v_c, v_used, v_its) = color_vals(dev_items, dev, d_o), color_vals(val_items, va, v_o)
        cg = {mu: score(with_w(dev_items, d_o, d_c, mu)) for mu in mu_grid}
        mu = max(cg, key=cg.get)
        res["val_v3+COLOR"] = met(with_w(val_items, v_o, v_c, mu), "v3+COLOR")
        chosen.update({"mu_color": mu, "dev_v3+COLOR": round(cg[mu], 4)})
        rel_imgs = {p for pr in (v_p if owl_chk is not None else []) if pr for p in pr[2]}
        col_imgs = {path for (path, prompt) in v_its if color_chk_conds(prompt)}
        res["color"] = {"val_queries_with_color_cond": sum(1 for it in val_items if color_chk_conds(prompts[it["q"]])),
                        "val_queries_decided": v_used,
                        "val_images_for_color": len(col_imgs), "extra_images_beyond_relation": len(col_imgs - rel_imgs),
                        "owl_new_forwards": color_chk.owl.n_forward - n0,
                        "owl_sec_per_image": round((color_chk.owl.sec - o0) / max(color_chk.owl.n_forward - n0, 1), 4),
                        "crops": color_chk.n_crops - c0,
                        "crop_sec_per_crop": round((color_chk.crop_sec - s0) / max(color_chk.n_crops - c0, 1), 4)}
        if owl_chk is not None:
            def swapped(items, orders, v, w):
                out = []
                for n, (order, sc) in enumerate(orders):
                    if sc[1] + w * v[2 * n + 1] > sc[0] + w * v[2 * n]:
                        order, sc = [order[1], order[0]] + order[2:], [sc[1], sc[0]] + sc[2:]
                    out.append((order, sc))
                return out
            v_oc = swapped(val_items, v_o, v_c, mu)
            res["val_v3+COLOR+OWL"] = met(owl_ranks(val_items, v_oc, owl_prep(val_items, va, v_oc), t, m), "v3+COLOR+OWL")
    res["chosen_on_dev"] = chosen
    res["per_query"] = per_q
    return res
