"""BLIP ITM 파인튜닝: 같은 물체 조합의 형제 이미지를 오답으로 두고, 질의마다 정답 이미지를 1위로 올리도록 학습한다.

- 학습 데이터: 제공된 train만 쓴다 (외부 데이터·LLM 증강 없음). 형제 그룹은 학습 데이터 안에서만 쓴다.
- 비전 인코더는 고정하고 이미지 특징을 한 번만 계산한다. 텍스트 인코더(이미지와의 교차 어텐션 포함)와 ITM 헤드만 학습.
- 손실: 질의 1개당 [정답 1 + 오답 N] 매칭 점수의 softmax 교차엔트로피 (1위 맞히기에 직접 맞춤).
- 평가(--final 아닐 때): 그룹 분할 학습 쪽에서 일부 그룹을 '개발용'으로 떼어 학습에 쓰지 않고, 재정렬 가중치는
  개발용에서 정한 뒤 그룹 분할 검증에서 한 번만 보고한다 (haiku.itm.run_itm_experiment 재사용).
  같은 조건에서 학습 전 BLIP과 학습 후 BLIP을 나란히 비교한다.

실행 (Colab / Kaggle GPU, 저장소 최상위):
    python -m haiku.itm_finetune --data-dir DATA --out-dir OUT            # 실험: 학습 → 검증 비교
    python -m haiku.itm_finetune --data-dir DATA --out-dir OUT --final    # 제출용: 4,000쌍 전체로 학습해 저장
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from haiku.data import load_rows
from haiku.itm import BLIP_ID, BlipITM, run_itm_experiment
from haiku.siblings import build_groups, parse_prompt

SIGLIP_ID = "google/siglip-base-patch16-384"


def wrap(model, proc, device):
    """학습한 모델을 BlipITM 인터페이스로 감싼다."""
    obj = BlipITM.__new__(BlipITM)
    obj.model, obj.proc, obj.device = model.eval(), proc, torch.device(device)
    obj.dtype = next(model.parameters()).dtype
    return obj


def negatives_for(q, group_of, members, pool, pool_list, n, rng):
    """오답: 같은 그룹(형제) 이미지를 먼저, 모자라면 학습 이미지에서 무작위로 채운다."""
    sib = [i for i in members[group_of[q]] if i != q and i in pool]
    rng.shuffle(sib)
    out = sib[:n]
    while len(out) < n:
        x = rng.choice(pool_list)
        if x != q and x not in out:
            out.append(x)
    return out


def retrieval_negatives(rows, ids, img_fn, txt_fn, lam_cond=3.2, K=10):
    """학습 질의마다 '추론 때 BLIP이 실제로 보게 될 후보' = SigLIP 검색 + 조건 재정렬 상위 K장 (정답 제외, 어려운 순).
    후보는 같은 학습 이미지 집합 안에서만 고른다."""
    from haiku.combo import prepare
    from haiku.rerank import PhraseBank, all_phrases, extract_conditions
    by_id = {r["id"]: r for r in rows}
    ids = list(ids)
    pos = {i: k for k, i in enumerate(ids)}
    prompts = {i: by_id[i]["prompt"] for i in ids}
    img = img_fn([by_id[i]["image_path"] for i in ids])
    txt = txt_fn([prompts[i] for i in ids])
    conds = {i: extract_conditions(prompts[i]) for i in ids}
    bank = PhraseBank(txt_fn)
    bank.add([p for i in ids for p in all_phrases(conds[i])])
    items, _ = prepare(ids, ids, txt, img, pos, conds, bank, lam_cond, K, None, None, prompts, 1)
    out = {it["q"]: [ids[j] for j in it["idxB"] if ids[j] != it["q"] and by_id[ids[j]]["prompt"] != prompts[it["q"]]]
           for it in items}
    hit = sum(it["gold"] in it["idxB"][:1] for it in items) / max(len(items), 1)
    print(f"검색 후보 오답 만들기: 질의 {len(items)}개, 후보 {K}장 (학습 이미지 안에서 1위 정답률 {hit:.3f})", flush=True)
    return out


@torch.no_grad()
def _vision_prefix(model, proc, paths, nv, device, bs=32):
    """이미지 쪽 앞 (전체 − nv)층까지의 출력을 한 번만 계산해 CPU에 fp16으로 보관 (마지막 nv층만 학습할 때)."""
    from PIL import Image
    vm = model.vision_model
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    out = None
    for s in range(0, len(paths), bs):
        ims = []
        for p in paths[s:s + bs]:
            with Image.open(p) as im:
                ims.append(im.convert("RGB"))
        px = proc.image_processor(images=ims, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = vm.embeddings(px)
            for lyr in vm.encoder.layers[:-nv]:
                o = lyr(h)
                h = o[0] if isinstance(o, tuple) else o
        h = h.to(torch.float16 if device.type == "cuda" else torch.float32).cpu()
        if out is None:
            out = torch.empty((len(paths),) + tuple(h.shape[1:]), dtype=h.dtype)
        out[s:s + len(h)] = h
    return out


def train_itm(rows, train_ids, args, device, neg_pool=None):
    """neg_pool: {질의 ID: [오답 후보 ID, 어려운 순]} 이 있으면 앞에서 n_hard개를 먼저 쓰고 나머지는 형제 오답으로 채운다."""
    from transformers import AutoProcessor, BlipForImageTextRetrieval
    by_id = {r["id"]: r for r in rows}
    parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
    group_of, members = build_groups(rows, parsed)
    torch.manual_seed(args.seed)
    model = BlipForImageTextRetrieval.from_pretrained(args.blip).to(device)
    proc = AutoProcessor.from_pretrained(args.blip)
    model.requires_grad_(False)
    model.text_encoder.requires_grad_(True)
    model.itm_head.requires_grad_(True)
    nv = int(getattr(args, "vision_layers", 0) or 0)                    # 이미지 쪽 마지막 nv층도 학습 (0이면 이미지 쪽 고정)
    vm = model.vision_model
    if nv:
        vm.encoder.layers[-nv:].requires_grad_(True)
        vm.post_layernorm.requires_grad_(True)
    vparams = [p for p in vm.parameters() if p.requires_grad]
    tparams = [p for p in model.parameters() if p.requires_grad and all(p is not q for q in vparams)]
    params = tparams + vparams
    groups = [{"params": tparams, "lr": args.lr}] + ([{"params": vparams, "lr": args.lr * 0.5}] if vparams else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 비전 특징 (고정): 학습 이미지 한 번만
    t0 = time.perf_counter()
    helper = wrap(model, proc, device)
    helper.dtype = torch.float16 if use_amp else torch.float32
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        if nv:
            feats = _vision_prefix(model, proc, [by_id[i]["image_path"] for i in train_ids], nv, device)
        else:
            feats = helper.image_embeds([by_id[i]["image_path"] for i in train_ids])
    pos = {i: k for k, i in enumerate(train_ids)}
    vis_sec = time.perf_counter() - t0
    print(f"비전 특징 {len(train_ids)}장 {vis_sec:.0f}초", flush=True)

    pool = set(train_ids)
    pool_list = sorted(pool)
    log, step = [], 0
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        rng = random.Random(args.seed * 1000 + ep)
        order = list(train_ids)
        rng.shuffle(order)
        model.train()
        losses, n_sib = [], 0
        for s in range(0, len(order), args.queries_per_step):
            qs = order[s:s + args.queries_per_step]
            texts, img_idx = [], []
            for q in qs:
                negs = negatives_for(q, group_of, members, pool, pool_list, args.negatives, rng)
                if neg_pool and q in neg_pool:
                    hard = [x for x in neg_pool[q] if x in pool][:getattr(args, "n_hard", 5)]
                    rest = [x for x in negs if x not in hard]
                    while len(hard) + len(rest) < args.negatives:
                        x = rng.choice(pool_list)
                        if x != q and x not in hard and x not in rest:
                            rest.append(x)
                    negs = hard + rest[:args.negatives - len(hard)]
                n_sib += sum(group_of[x] == group_of[q] for x in negs)
                texts += [by_id[q]["prompt"]] * (1 + len(negs))
                img_idx += [pos[q]] + [pos[x] for x in negs]
            tok = proc.tokenizer(texts, padding=True, truncation=True, max_length=72, return_tensors="pt").to(device)
            img = feats[img_idx].to(device)
            if nv:                                                      # 저장해 둔 앞쪽 층 출력 → 학습하는 마지막 층들
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    h = img.float() if not use_amp else img
                    for lyr in vm.encoder.layers[-nv:]:
                        o = lyr(h)
                        h = o[0] if isinstance(o, tuple) else o
                    img = vm.post_layernorm(h)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                att = torch.ones(img.shape[:-1], dtype=torch.long, device=device)
                h = model.text_encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                       encoder_hidden_states=img.to(torch.float16 if use_amp else torch.float32),
                                       encoder_attention_mask=att).last_hidden_state
                logit = model.itm_head(h[:, 0, :]).float()
            match = (logit[:, 1] - logit[:, 0]).view(len(qs), 1 + args.negatives)
            loss = torch.nn.functional.cross_entropy(match, torch.zeros(len(qs), dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        log.append({"epoch": ep + 1, "loss": round(sum(losses) / len(losses), 4),
                    "sibling_negatives_ratio": round(n_sib / (len(order) * args.negatives), 3)})
        print(f"  epoch {ep + 1}/{args.epochs} loss {log[-1]['loss']:.4f} | 오답 중 형제 비율 {log[-1]['sibling_negatives_ratio']}",
              flush=True)
    model.eval()
    return model, proc, {"steps": step, "train_sec": round(time.perf_counter() - t0, 1), "epochs": log,
                         "vision_sec": round(vis_sec, 1), "trainable_params": sum(p.numel() for p in params)}


def siglip_embeddings(rows, ids, model_id, device, smoke_dir=None):
    from PIL import Image
    from torch.nn.functional import normalize
    from transformers import AutoModel, AutoProcessor
    src = smoke_dir or model_id
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = AutoModel.from_pretrained(src, dtype=dtype, local_files_only=bool(smoke_dir)).to(device).eval()
    proc = AutoProcessor.from_pretrained(src, local_files_only=bool(smoke_dir))
    by_id = {r["id"]: r for r in rows}
    feats = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o

    @torch.inference_mode()
    def enc_txt(t):
        out = []
        for s in range(0, len(t), 256):
            tk = proc(text=t[s:s + 256], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            out.append(normalize(feats(net.get_text_features(**tk)).float(), dim=-1))
        return torch.cat(out)

    with torch.inference_mode():
        img = []
        for s in range(0, len(ids), 32):
            ims = [Image.open(by_id[i]["image_path"]).convert("RGB") for i in ids[s:s + 32]]
            px = proc(images=ims, return_tensors="pt").pixel_values.to(device, dtype)
            img.append(normalize(feats(net.get_image_features(pixel_values=px)).float(), dim=-1))
    return torch.cat(img), enc_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("splits/split_group_seed42.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--blip", default=BLIP_ID)
    ap.add_argument("--siglip", default=SIGLIP_ID)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--queries-per-step", type=int, default=8)
    ap.add_argument("--negatives", type=int, default=7)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--eval-k", type=int, nargs="+", default=[2, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="4,000쌍 전체로 학습해 저장 (평가 없음)")
    ap.add_argument("--smoke", action="store_true", help="작은 점검 (CPU 가능)")
    ap.add_argument("--smoke-siglip-dir", default=None)
    ap.add_argument("--neg-mode", choices=["sibling", "retrieval"], default="sibling",
                    help="retrieval: 추론 때 BLIP이 보는 후보(SigLIP+조건 재정렬 상위 10장)를 오답으로 먼저 사용")
    ap.add_argument("--n-hard", type=int, default=5)
    ap.add_argument("--vision-layers", type=int, default=0, help="이미지 쪽 마지막 몇 층을 함께 학습할지 (0 = 고정)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.data_dir)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.final:
        train_ids = [r["id"] for r in rows]
        neg_pool = None
        if args.neg_mode == "retrieval":
            from haiku.combo import _siglip
            img_fn, txt_fn = _siglip(args.siglip, device)
            neg_pool = retrieval_negatives(rows, train_ids, img_fn, txt_fn)
            del img_fn, txt_fn
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model, proc, info = train_itm(rows, train_ids, args, device, neg_pool)
        save = args.out_dir / "blip_itm_ft"
        model.half().save_pretrained(save)
        proc.save_pretrained(save)
        (args.out_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print("저장:", save)
        return

    # 개발용 그룹을 학습 분할에서 떼어 낸다 (재정렬 가중치 결정용, 학습에는 쓰지 않음)
    parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
    group_of, _ = build_groups(rows, parsed)
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    groups = sorted({group_of[i] for i in tr})
    random.Random(args.seed).shuffle(groups)
    dev_groups = set(groups[:max(1, int(len(groups) * args.dev_frac))])
    dev = [i for i in tr if group_of[i] in dev_groups]
    ft = [i for i in tr if group_of[i] not in dev_groups]
    if args.smoke:
        ft, dev, va = ft[:48], dev[:24], va[:24]
        args.epochs, args.negatives = 1, 3
    print(f"ITM 학습 {len(ft)} / 개발 {len(dev)} / 검증 {len(va)} (그룹이 겹치지 않음)", flush=True)

    img, enc_txt = siglip_embeddings(rows, dev + va, args.siglip, device, args.smoke_siglip_dir)
    sub = {"train_ids": dev, "valid_ids": va}
    pairs_csv, contrast_csv = "reports/siblings/pairs.csv", "reports/siblings/contrast_phrases.csv"
    K = tuple(args.eval_k)
    results = {"sizes": {"itm_train": len(ft), "dev": len(dev), "valid": len(va)}, "args": vars(args)}

    base = BlipITM(args.blip, device)
    results["before"] = run_itm_experiment(rows, sub, pairs_csv, contrast_csv, img, enc_txt, base, K_list=K,
                                           per_cat=200)
    del base
    model, proc, info = train_itm(rows, ft, args, device)
    results["train"] = info
    results["after"] = run_itm_experiment(rows, sub, pairs_csv, contrast_csv, img, enc_txt, wrap(model, proc, device),
                                          K_list=K, per_cat=200)
    (args.out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
    print(summary(results))


def summary(res):
    L = ["", "===== BLIP ITM 파인튜닝 결과 (검증 = 그룹 분할 검증, 가중치 = 개발 그룹에서 결정) ====="]
    L.append(f"크기: {res['sizes']} | 학습 {res['train']['train_sec']}초, {res['train']['epochs']}")
    for K, rb in res["before"]["by_K"].items():
        ra = res["after"]["by_K"][K]
        L.append(f"\n[Top-{K}] 검증 Hit@{K} {rb['val_hit@K']}")
        L.append(f"{'방식':10s} {'학습 전 점수':>10s} {'R@1':>7s} {'1위로 고침':>8s} {'1위 놓침':>8s} | {'학습 후 점수':>10s} {'R@1':>7s} {'1위로 고침':>8s} {'1위 놓침':>8s}")
        for m in ("A", "B", "B_gate", "C"):
            b, a = rb[f"val_{m}"], ra[f"val_{m}"]
            L.append(f"{m:10s} {b['score']:>10.4f} {b['recall@1']:>7.4f} {b['wrong_to_right@1']:>8d} {b['right_to_wrong@1']:>8d} | "
                     f"{a['score']:>10.4f} {a['recall@1']:>7.4f} {a['wrong_to_right@1']:>8d} {a['right_to_wrong@1']:>8d}")
        L.append(f"선택된 값 (개발): 학습 전 {rb['chosen']}, 학습 후 {ra['chosen']}")
    for key in ("probe_original_BLIP", "probe_phrase_BLIP"):
        b, a = res["before"][key], res["after"][key]
        L.append(f"\n2×2 점검 {key} (두 문장 모두 자기 이미지 선택 / 전체): " + ", ".join(
            f"{c}: {b[c]['both_ratio']}→{a.get(c, {}).get('both_ratio', '-')}" for c in sorted(b)))
    return "\n".join(L)


if __name__ == "__main__":
    main()
