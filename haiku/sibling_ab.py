"""A/B 실험: 학습 배치 구성만 바꿨을 때 형제 오답을 더 잘 구별하는가.

A = 무작위 배치, B = 같은 물체 집합이면서 물체 조건이 명시적으로 다른 쌍(학습 분할 안에서만)을 배치마다 일부 넣고
나머지는 일반 샘플로 채운 배치. 그 밖의 조건은 모두 같다.

같게 맞춘 것:
  - 모델 초기값(같은 사전학습 가중치, 같은 torch seed), 학습 대상(기본: 출력 헤드만 = 스타터 키트 train.py와 같음)
  - 배치 크기·에포크·스텝 수·학습률, 해상도(224), 증강 없음(무작위 반전·색 변형·크롭 없음)
  - 에포크마다 쓰는 이미지 ID 집합과 노출 횟수(모든 학습 ID가 에포크당 정확히 한 번)
  - 검증 질의·후보 목록(검증 이미지 전체), 평가식(공식 점수)
손실: SigLIP sigmoid 손실(transformers SiglipModel, return_loss=True). 배치 안의 모든 다른 이미지–텍스트 조합이 오답으로
쓰이므로 배치 구성만 바꿔도 학습 신호가 달라진다.

실행 (Colab / Kaggle, GPU 1장, 저장소 최상위):
    python -m haiku.sibling_ab --data-dir /content/work/haikuthon-train \
        --out-dir /content/drive/MyDrive/HAIKUTHON/results/exp_sibling_ab --cache-dir /content/work/cache
작은 점검 (CPU 가능, 다운로드 없음, 무작위 초기화 소형 모델):
    python -m haiku.sibling_ab --smoke --data-dir haikuthon-train --processor-dir <스타터키트>/sample_submission/models
"""

import argparse
import csv
import json
import math
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from haiku.data import load_rows
from haiku.metrics import metrics_from_ranks
from haiku.retrieval import ranks_of_gold, read_image
from haiku.siblings import (batch_pair_stats, build_groups, build_pairs, compare, fmt_diffs, parse_prompt,
                            random_batches, sibling_batches)

BASE_MODEL = "google/siglip-base-patch16-224"


# ----------------------------------------------------------------------------- 데이터 준비
def prepare_pairs(rows):
    parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
    group_of, members = build_groups(rows, parsed)
    pairs = build_pairs(rows, parsed, members)
    return parsed, group_of, pairs


def cache_inputs(rows, processor, cache_path=None, workers=2):
    """이미지는 증강 없이 한 번만 전처리해 fp16 텐서로 보관한다 (A/B가 완전히 같은 입력을 쓰게 하려는 목적)."""
    ids = [r["id"] for r in rows]
    if cache_path and Path(cache_path).exists():
        c = torch.load(cache_path)
        if c["ids"] == ids:
            print(f"전처리 캐시 사용: {cache_path}")
            return c["pixels"], c["input_ids"]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return processor(images=read_image(rows[i]["image_path"]), return_tensors="pt").pixel_values[0].half()

    t0 = time.time()
    dl = torch.utils.data.DataLoader(DS(), batch_size=64, num_workers=workers)
    pixels = torch.cat(list(dl))
    input_ids = processor(text=[r["prompt"] for r in rows], padding="max_length", truncation=True,
                          return_tensors="pt")["input_ids"]
    print(f"전처리 {len(rows)}장 {time.time() - t0:.0f}초, pixels {tuple(pixels.shape)}")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "pixels": pixels, "input_ids": input_ids}, cache_path)
    return pixels, input_ids


# ----------------------------------------------------------------------------- 모델
def build_model(args, device):
    from transformers import SiglipConfig, SiglipModel
    if args.smoke:
        cfg = SiglipConfig(
            text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                         "vocab_size": 32000, "max_position_embeddings": 64},
            vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                           "image_size": 224, "patch_size": 16})
        net = SiglipModel(cfg)
    else:
        net = SiglipModel.from_pretrained(args.base_model)
    net.requires_grad_(False)
    net.text_model.head.requires_grad_(True)
    net.vision_model.head.requires_grad_(True)
    if args.trainable == "last2":
        for layers in (net.vision_model.encoder.layers[-2:], net.text_model.encoder.layers[-2:]):
            layers.requires_grad_(True)
    return net.to(device)


def feats(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


@torch.inference_mode()
def evaluate(net, pixels, input_ids, q_idx, g_idx, device, bs=64):
    net.eval()
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        iv = torch.cat([normalize(feats(net.get_image_features(pixel_values=pixels[g_idx[s:s + bs]].to(device).float())).float(), dim=-1)
                        for s in range(0, len(g_idx), bs)])
        tv = torch.cat([normalize(feats(net.get_text_features(input_ids=input_ids[q_idx[s:s + bs]].to(device))).float(), dim=-1)
                        for s in range(0, len(q_idx), bs)])
    pos = {g: k for k, g in enumerate(g_idx)}
    gold = [pos[q] for q in q_idx]
    ranks = ranks_of_gold(tv, iv, gold)
    top1 = (tv @ iv.T).argmax(1).cpu().tolist()
    return ranks, [g_idx[t] for t in top1]


def train_arm(arm, args, seed, pixels, input_ids, index_of, train_ids, clear_train, group_of, pair_index, device):
    torch.manual_seed(seed)
    net = build_model(args, device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    log, t0, step = [], time.time(), 0
    for ep in range(args.epochs):
        rng = random.Random(seed * 1000 + ep)
        if arm == "A":
            batches, chosen = random_batches(train_ids, args.bs, rng), []
        else:
            batches, chosen = sibling_batches(train_ids, clear_train, args.bs, args.pairs_per_batch, rng)
        st = batch_pair_stats(batches, group_of, pair_index)
        st.update(epoch=ep, chosen_pairs=len(chosen))
        net.train()
        losses = []
        for b in batches:
            idx = [index_of[i] for i in b]
            out = net(input_ids=input_ids[idx].to(device), pixel_values=pixels[idx].to(device).float(), return_loss=True)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            losses.append(out.loss.item())
            step += 1
        st["mean_loss"] = round(sum(losses) / len(losses), 4)
        log.append(st)
        print(f"  [{arm} seed{seed}] epoch {ep + 1}/{args.epochs} loss {st['mean_loss']:.4f} "
              f"| 배치 안 명확한 형제 쌍 {st['clear_pairs_in_batches']} ({st['clear_pairs_per_batch']}/배치)")
    return net, {"steps": step, "train_sec": round(time.time() - t0, 1), "epochs": log,
                 "trainable_params": sum(p.numel() for p in params)}


# ----------------------------------------------------------------------------- 결과 정리
def subset_metrics(ranks, mask):
    r = np.asarray(ranks)[np.asarray(mask, bool)]
    return metrics_from_ranks(r) if len(r) else None


def summarize_eval(ranks, sib_mask):
    full, sub = metrics_from_ranks(ranks), subset_metrics(ranks, sib_mask)
    keep = ("n_queries", "score", "recall@1", "ndcg@5", "hit@5", "hit@20", "rank_dist")
    return {"all": {k: full[k] for k in keep}, "with_sibling": {k: sub[k] for k in keep} if sub else None}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{statistics.mean(vals):.4f}" + (f" ± {statistics.stdev(vals):.4f}" if len(vals) > 1 else "")


def save_cases(cases, rows_by_id, parsed, images_dir, path, thumb=200):
    from PIL import Image, ImageDraw
    if not cases:
        return
    pad, text_h = 8, 92
    W, H = 2 * thumb + 3 * pad, len(cases) * (thumb + text_h + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for kind, q, wrong, ra, rb in cases:
        for k, img in enumerate((q, wrong)):
            with Image.open(Path(images_dir) / f"{img}.png") as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (pad + k * (thumb + pad), y))
        same_group = parsed[q]["objects"].keys() == parsed[wrong]["objects"].keys()
        diff = fmt_diffs(compare(parsed[q], parsed[wrong]), "explicit") if same_group else "형제 아님"
        t = rows_by_id[q]["prompt"].replace("\n", " ")
        lines = [f"[{kind}] query {q} (left=gold) | top-1 wrong {wrong} (right) | rank A={ra} B={rb}",
                 f"diff(gold vs wrong, 설명 비교): {diff[:95]}", t[:85], t[85:170], t[170:255]]
        for k, line in enumerate(lines):
            draw.text((pad, y + thumb + 3 + k * 14), line, fill="black")
        y += thumb + text_h + pad
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("haikuthon-train"))
    ap.add_argument("--split", type=Path, default=Path("splits/split_group_seed42.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/exp_sibling_ab"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--processor-dir", default=None, help="전처리 설정 위치 (기본: --base-model)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pairs-per-batch", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trainable", choices=["heads", "last2"], default="heads")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="작은 점검: 소형 무작위 모델, 학습 64 / 검증 32장, 1에포크")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="첫 seed의 A·B 모델을 저장할 폴더 (재정렬과 합친 조합 평가용, 학습 분할로만 학습한 모델)")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor_dir or args.base_model,
                                              local_files_only=bool(args.processor_dir))

    # --- 분할·쌍·누수 점검
    rows = load_rows(args.data_dir)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids, valid_ids = split["train_ids"], split["valid_ids"]
    parsed, group_of, pairs = prepare_pairs(rows)
    if args.smoke:
        rng = random.Random(0)
        tg = sorted({group_of[i] for i in train_ids})
        vg = sorted({group_of[i] for i in valid_ids})
        tg, vg = set(rng.sample(tg, 12)), set(rng.sample(vg, 6))
        train_ids = [i for i in train_ids if group_of[i] in tg][:64]
        valid_ids = [i for i in valid_ids if group_of[i] in vg][:32]
    tr, va = set(train_ids), set(valid_ids)
    assert not tr & va, "학습·검증 ID가 겹칩니다."
    crossing = {group_of[i] for i in tr} & {group_of[i] for i in va}
    pair_index = {(p["a"], p["b"]): p for p in pairs}
    clear_train = [(p["a"], p["b"]) for p in pairs if p["clear"] and p["a"] in tr and p["b"] in tr]
    assert all(a in tr and b in tr for a, b in clear_train)
    rows_by_id = {r["id"]: r for r in rows}
    use_rows = [rows_by_id[i] for i in sorted(tr | va, key=int)]
    index_of = {r["id"]: k for k, r in enumerate(use_rows)}
    print(f"분할 {args.split.name}: 학습 {len(train_ids)} / 검증 {len(valid_ids)}, 겹치는 그룹 {len(crossing)}, "
          f"학습 안 명확한 형제 쌍 {len(clear_train)}")

    cache = None if args.smoke or not args.cache_dir else args.cache_dir / f"inputs_{args.base_model.split('/')[-1]}_{len(use_rows)}.pt"
    pixels, input_ids = cache_inputs(use_rows, processor, cache, args.workers)

    q_ids = list(valid_ids)
    q_idx = [index_of[i] for i in q_ids]
    by_group = {}
    for i in valid_ids:
        by_group.setdefault(group_of[i], set()).add(i)
    n_sib = [len(by_group[group_of[i]] - {i}) for i in q_ids]
    sib_mask = [n > 0 for n in n_sib]
    eval_info = {"n_queries": len(q_ids), "n_gallery": len(valid_ids),
                 "mean_siblings_per_query": round(sum(n_sib) / len(n_sib), 3), "queries_with_sibling": sum(sib_mask)}
    print("검증:", eval_info)

    # --- 사전학습 모델(학습 전) 기준
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "split": args.split.name, "eval": eval_info,
               "leak_check": {"overlap_ids": len(tr & va), "groups_crossing": len(crossing),
                              "clear_pairs_outside_train": 0}, "runs": []}
    net0 = build_model(args, device)
    r0, _ = evaluate(net0, pixels, input_ids, q_idx, q_idx, device)
    results["pretrained"] = summarize_eval(r0, sib_mask)
    del net0
    print("학습 전:", results["pretrained"]["all"]["score"])

    per_query = {}
    for seed in args.seeds:
        for arm in ("A", "B"):
            net, info = train_arm(arm, args, seed, pixels, input_ids, index_of, train_ids, clear_train, group_of,
                                  pair_index, device)
            ranks, top1 = evaluate(net, pixels, input_ids, q_idx, q_idx, device)
            per_query[(seed, arm)] = (ranks, [use_rows[t]["id"] for t in top1])
            ev = summarize_eval(ranks, sib_mask)
            results["runs"].append({"seed": seed, "arm": arm, **info, **ev})
            print(f"  [{arm} seed{seed}] 전체 {ev['all']['score']:.4f} / 형제 오답 있는 질의 "
                  f"{(ev['with_sibling'] or {}).get('score', float('nan')):.4f}")
            if args.save_dir and seed == args.seeds[0] and not args.smoke:
                out = args.save_dir / f"{arm}_seed{seed}"
                net.half().save_pretrained(out)
                processor.save_pretrained(out)
                print(f"  저장: {out} (그룹 분할 학습 쪽으로만 학습)", flush=True)
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- A/B 비교 (같은 seed끼리 짝지어 비교)
    flips = []
    for seed in args.seeds:
        ra, rb = np.asarray(per_query[(seed, "A")][0]), np.asarray(per_query[(seed, "B")][0])
        flips.append({"seed": seed, "A_wrong_to_B_right@1": int(((ra > 1) & (rb == 1)).sum()),
                      "A_right_to_B_wrong@1": int(((ra == 1) & (rb > 1)).sum()),
                      "A_wrong_to_B_right@1_with_sibling": int(((ra > 1) & (rb == 1) & np.asarray(sib_mask)).sum()),
                      "A_right_to_B_wrong@1_with_sibling": int(((ra == 1) & (rb > 1) & np.asarray(sib_mask)).sum())})
    results["flips"] = flips

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_query_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "query_id", "rank", "top1_id", "n_siblings_in_gallery"])
        for (seed, arm), (ranks, top1) in per_query.items():
            for q, r, t, n in zip(q_ids, ranks, top1, n_sib):
                w.writerow([seed, arm, q, int(r), t, n])

    s0 = args.seeds[0]
    ra, ta = per_query[(s0, "A")]
    rb, tb = per_query[(s0, "B")]
    improved = [("개선", q, ta[k], int(ra[k]), int(rb[k])) for k, q in enumerate(q_ids) if ra[k] > 1 and rb[k] == 1]
    worsened = [("악화", q, tb[k], int(ra[k]), int(rb[k])) for k, q in enumerate(q_ids) if ra[k] == 1 and rb[k] > 1]
    rng = random.Random(0)
    cases = rng.sample(improved, min(6, len(improved))) + rng.sample(worsened, min(6, len(worsened)))
    if not args.smoke:
        save_cases(cases, rows_by_id, parsed, args.data_dir / "images", args.out_dir / "cases.jpg")
    results["cases_seed"] = s0
    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (args.out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render(results, args), encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def render(res, args):
    L = []
    a = L.append
    a(f"# 형제 배치 A/B 실험 ({'작은 점검(smoke)' if args.smoke else '전체'})\n")
    a(f"- 분할: `{res['split']}` · 검증: 질의 {res['eval']['n_queries']}, 후보 {res['eval']['n_gallery']} (검증 이미지 전체), "
      f"질의당 형제 오답 평균 {res['eval']['mean_siblings_per_query']}, 형제 오답이 있는 질의 {res['eval']['queries_with_sibling']}")
    a(f"- 설정: 모델 `{args.base_model}`{' (smoke: 무작위 소형 모델)' if args.smoke else ''}, 학습 대상 {args.trainable}, "
      f"에포크 {args.epochs}, 배치 {args.bs}, lr {args.lr}, B의 배치당 비교 쌍 최대 {args.pairs_per_batch}, seed {args.seeds}, 증강 없음, 224px")
    a(f"- 누수 점검: {res['leak_check']}")
    a(f"- 학습 전 모델: 전체 {res['pretrained']['all']['score']:.4f}, 형제 오답 있는 질의 "
      f"{(res['pretrained']['with_sibling'] or {}).get('score', float('nan')):.4f}\n")
    a("| 배치 | 전체 공식 점수 | 형제 오답 있는 질의의 공식 점수 | Recall@1 | Recall@5 | 학습 스텝 | 배치당 명확한 형제 쌍(마지막 에포크) |")
    a("|---|---|---|---|---|---|---|")
    for arm in ("A", "B"):
        runs = [r for r in res["runs"] if r["arm"] == arm]
        a(f"| {arm} | {mean_std([r['all']['score'] for r in runs])} | "
          f"{mean_std([(r['with_sibling'] or {}).get('score') for r in runs])} | "
          f"{mean_std([r['all']['recall@1'] for r in runs])} | {mean_std([r['all']['hit@5'] for r in runs])} | "
          f"{runs[0]['steps']} | {mean_std([r['epochs'][-1]['clear_pairs_per_batch'] for r in runs])} |")
    a("\n값은 seed별 평균 ± 표준편차. Recall@5 = 정답이 5위 안에 든 비율.\n")
    a("| seed | A 오답 → B 1위 정답 | A 1위 정답 → B 놓침 | (형제 오답 있는 질의만) 개선 | 악화 |")
    a("|---|---|---|---|---|")
    for f in res["flips"]:
        a(f"| {f['seed']} | {f['A_wrong_to_B_right@1']} | {f['A_right_to_B_wrong@1']} | "
          f"{f['A_wrong_to_B_right@1_with_sibling']} | {f['A_right_to_B_wrong@1_with_sibling']} |")
    a("\n배치 구성 기록 (에포크별 실제 배치 안의 형제 쌍):\n")
    for r in res["runs"]:
        a(f"- {r['arm']} seed{r['seed']}: " + "; ".join(
            f"ep{e['epoch']} 형제쌍 {e['sibling_pairs_in_batches']}, 명확 {e['clear_pairs_in_batches']}, 종류 {e['categories']}"
            for e in r["epochs"]))
    a(f"\n대표 개선·악화 사례: `cases.jpg` (seed {res['cases_seed']}, 왼쪽 정답 이미지 / 오른쪽 1위 오답). "
      "사례 설명의 '설명 비교'는 분석 표시용이며 검색 점수 계산에는 쓰지 않는다.")
    a("\n해석 주의: 그룹 분할 검증은 처음 보는 물체 조합에 대한 일반화도 함께 잰다. seed 간 표준편차보다 작은 차이는 우연일 수 있다.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
