"""최종 조합 평가: SigLIP 검색 → 조건 재정렬(B) → BLIP ITM 재정렬.

- 같은 질의·같은 후보에서 A(검색만) / Bc(조건 재정렬) / Bc+ITM(상위 k개를 BLIP으로 다시 정렬) / Bc+ITM_gate(1·2위 박빙만)를 비교한다.
- 조건 재정렬 λ는 R1에서 개발 데이터로 정한 값을 쓰고, ITM의 k·비중·박빙 기준은 개발 데이터에서 정한 뒤 검증에서 한 번만 보고한다.
- BLIP은 학습 전 모델과, ITM 학습 분할로만 학습한 모델을 각각 평가할 수 있다
  (itm_finetune과 같은 seed로 개발 그룹을 떼므로 [3/3]과 같은 분할이다).
- 순위는 현재 질의 문장과 그 질의의 후보 이미지만으로 계산한다.

실행 (Colab / Kaggle GPU, 저장소 최상위):
    python -m haiku.combo --data-dir DATA --out-dir OUT [--lam-cond 3.2] [--itm-dir 학습한_BLIP_폴더] [--train-itm]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from haiku.data import load_rows
from haiku.itm import BLIP_ID, BlipITM
from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores
from haiku.siblings import build_groups, parse_prompt

SIGLIP_ID = "google/siglip-base-patch16-384"


def _score(ranks):
    return sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in ranks) / len(ranks)


def _metrics(ranks, base):
    return {"score": round(_score(ranks), 4), "recall@1": round(sum(r == 1 for r in ranks) / len(ranks), 4),
            "wrong_to_right@1": sum(b > 1 and r == 1 for b, r in zip(base, ranks)),
            "right_to_wrong@1": sum(b == 1 and r > 1 for b, r in zip(base, ranks))}


def prepare(q_ids, g_ids, txt, img, pos, conds, bank, lam_cond, K, itm, feats, prompts, k_max):
    G = img[[pos[g] for g in g_ids]]
    gpos = {g: k for k, g in enumerate(g_ids)}
    items = []
    for q in q_ids:
        s = txt[pos[q]] @ G.T
        top = s.topk(min(K, len(g_ids)))
        gold = gpos[q]
        M = condition_margins(conds[q], bank, G[top.indices])
        newb = rerank_scores(top.values, conds[q], M, "B", lam_cond, {}) if lam_cond > 0 else top.values
        o = newb.argsort(descending=True)
        items.append({"q": q, "gold": gold, "base_rank": int((s >= s[gold]).sum()),
                      "idxA": top.indices.tolist(), "idxB": top.indices[o].tolist(), "sB": newb[o].float().cpu()})
    texts, idx = [], []
    for it in items:
        texts += [prompts[it["q"]]] * k_max
        idx += [pos[g_ids[j]] for j in it["idxB"][:k_max]]
    t0 = time.perf_counter()
    sc = itm.score_idx(texts, idx, feats).view(len(items), k_max) if itm is not None else None
    sec = time.perf_counter() - t0
    for n, it in enumerate(items):
        it["itm"] = sc[n] if sc is not None else None
        it["gapB"] = float(it["sB"][0] - it["sB"][1]) if len(it["sB"]) > 1 else 1.0
    return items, {"itm_pairs": len(texts), "itm_sec": round(sec, 2)}


def ranks_of(items, stage, k=0, lam=0.0, tau=None):
    out = []
    for it in items:
        order = list(it["idxA"] if stage == "A" else it["idxB"])
        if stage == "ITM" and (tau is None or it["gapB"] < tau):
            new = it["sB"][:k] + lam * it["itm"][:k]
            head = [order[j] for j in new.argsort(descending=True).tolist()]
            order = head + order[k:]
        out.append(order.index(it["gold"]) + 1 if it["gold"] in order else it["base_rank"])
    return out


def run_combo(rows, dev_ids, val_ids, siglip_img_fn, siglip_txt_fn, itm, lam_cond, K=10, k_list=(2, 3, 5),
              lam_grid=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0), tau_q=(0.2, 0.4, 0.6, 0.8, 1.01)):
    by_id = {r["id"]: r for r in rows}
    ids = list(dev_ids) + list(val_ids)
    pos = {i: k for k, i in enumerate(ids)}
    prompts = {i: by_id[i]["prompt"] for i in ids}
    img = siglip_img_fn([by_id[i]["image_path"] for i in ids])
    txt = siglip_txt_fn([prompts[i] for i in ids])
    conds = {i: extract_conditions(prompts[i]) for i in ids}
    bank = PhraseBank(siglip_txt_fn)
    bank.add([p for i in ids for p in all_phrases(conds[i])])
    t0 = time.perf_counter()
    feats = itm.image_embeds([by_id[i]["image_path"] for i in ids])
    out = {"blip_img_per_sec": round(len(ids) / (time.perf_counter() - t0), 1)}
    k_max = max(k_list)
    dev, dev_t = prepare(dev_ids, dev_ids, txt, img, pos, conds, bank, lam_cond, K, itm, feats, prompts, k_max)
    val, val_t = prepare(val_ids, val_ids, txt, img, pos, conds, bank, lam_cond, K, itm, feats, prompts, k_max)
    out["itm_pairs_per_sec"] = round((dev_t["itm_pairs"] + val_t["itm_pairs"]) / max(1e-6, dev_t["itm_sec"] + val_t["itm_sec"]), 1)
    base_dev, base_val = ranks_of(dev, "A"), ranks_of(val, "A")
    # 개발 데이터에서 k, λ, τ 선택
    best = None
    grid = {}
    for k in k_list:
        for lam in lam_grid:
            sc = _score(ranks_of(dev, "ITM", k, lam))
            grid[f"k{k}_lam{lam}"] = round(sc, 4)
            if best is None or sc > best[0]:
                best = (sc, k, lam)
    _, k_best, lam_best = best
    gaps = sorted(it["gapB"] for it in dev)
    taus = sorted({gaps[min(len(gaps) - 1, int(q * len(gaps)))] if q <= 1 else 1e9 for q in tau_q})
    tsc = {t: _score(ranks_of(dev, "ITM", k_best, lam_best, t)) for t in taus}
    tau_best = max(tsc, key=tsc.get)
    out["dev"] = {"A": round(_score(base_dev), 4), "Bc": round(_score(ranks_of(dev, "Bc")), 4), "grid": grid,
                  "chosen": {"k": k_best, "lam": lam_best, "tau": tau_best if tau_best < 1e8 else None}}
    out["val"] = {"A": _metrics(base_val, base_val), "Bc": _metrics(ranks_of(val, "Bc"), base_val),
                  "Bc+ITM": _metrics(ranks_of(val, "ITM", k_best, lam_best), base_val),
                  "Bc+ITM_gate": _metrics(ranks_of(val, "ITM", k_best, lam_best, tau_best), base_val)}
    for k in k_list:   # 참고: 각 k에서 개발 최적 λ로 검증
        lam_k = max(lam_grid, key=lambda l: grid[f"k{k}_lam{l}"])
        out["val"][f"Bc+ITM_k{k}"] = _metrics(ranks_of(val, "ITM", k, lam_k), base_val)
    out["val_gate_ratio"] = round(sum(it["gapB"] < tau_best for it in val) / len(val), 4)
    out["sizes"] = {"dev": len(dev_ids), "val": len(val_ids)}
    return out


def _siglip(model_id, device):
    from PIL import Image
    from torch.nn.functional import normalize
    from transformers import AutoModel, AutoProcessor
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    f = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o

    @torch.inference_mode()
    def img_fn(paths):
        out = []
        for s in range(0, len(paths), 32):
            px = proc(images=[Image.open(p).convert("RGB") for p in paths[s:s + 32]], return_tensors="pt").pixel_values
            out.append(normalize(f(net.get_image_features(pixel_values=px.to(device, dtype))).float(), dim=-1))
        return torch.cat(out)

    @torch.inference_mode()
    def txt_fn(t):
        out = []
        for s in range(0, len(t), 256):
            tk = proc(text=t[s:s + 256], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            out.append(normalize(f(net.get_text_features(**tk)).float(), dim=-1))
        return torch.cat(out)
    return img_fn, txt_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("splits/split_group_seed42.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--siglip", default=SIGLIP_ID)
    ap.add_argument("--lam-cond", type=float, default=3.2)
    ap.add_argument("--train-itm", action="store_true", help="ITM 학습 분할로 BLIP을 학습해 함께 평가 ([3/3]과 같은 분할)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.data_dir)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
    group_of, _ = build_groups(rows, parsed)
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    groups = sorted({group_of[i] for i in tr})
    random.Random(args.seed).shuffle(groups)
    dev_groups = set(groups[:max(1, int(len(groups) * args.dev_frac))])
    dev = [i for i in tr if group_of[i] in dev_groups]
    ft = [i for i in tr if group_of[i] not in dev_groups]
    print(f"개발 {len(dev)} / 검증 {len(va)} (ITM 학습용 {len(ft)}), 조건 재정렬 λ={args.lam_cond}", flush=True)
    img_fn, txt_fn = _siglip(args.siglip, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = {"lam_cond": args.lam_cond}
    res["blip_base"] = run_combo(rows, dev, va, img_fn, txt_fn, BlipITM(BLIP_ID, device), args.lam_cond)
    print("BLIP(학습 전) 조합:", json.dumps(res["blip_base"]["val"], ensure_ascii=False), flush=True)
    if args.train_itm:
        from haiku.itm_finetune import train_itm, wrap
        ns = argparse.Namespace(blip=BLIP_ID, seed=args.seed, lr=2e-5, epochs=args.epochs, queries_per_step=8, negatives=7)
        model, proc, info = train_itm(rows, ft, ns, device)
        res["train_info"] = info
        res["blip_ft"] = run_combo(rows, dev, va, img_fn, txt_fn, wrap(model, proc, device), args.lam_cond)
    (args.out_dir / "combo_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(summary(res))


def summary(res):
    L = ["", f"===== 최종 조합 평가 (검증 = 그룹 분할 검증, 값은 개발 그룹에서 결정, 조건 재정렬 λ={res['lam_cond']}) ====="]
    for key, name in (("blip_base", "BLIP 학습 전"), ("blip_ft", "BLIP 학습 후")):
        if key not in res:
            continue
        r = res[key]
        L.append(f"\n■ {name} | 개발에서 고른 값 {r['dev']['chosen']} | BLIP {r['blip_img_per_sec']}장/초, ITM {r['itm_pairs_per_sec']}쌍/초 | 박빙 비율 {r['val_gate_ratio']}")
        for m, v in r["val"].items():
            L.append(f"  {m:14s} 점수 {v['score']:.4f}  R@1 {v['recall@1']:.4f}  1위로 고침 {v['wrong_to_right@1']:3d}  1위 놓침 {v['right_to_wrong@1']:3d}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
