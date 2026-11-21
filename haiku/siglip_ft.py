"""SigLIP 검색 모델을 우리 학습 데이터로 미세조정 (형제 묶음 배치 + 문장 최소 쌍).

지금까지 검색·조건 재정렬에 쓰는 SigLIP은 사전학습 그대로였다 (BLIP만 학습). 테스트에서는 정답이 최종 5장에 드는 비율이
61%뿐이라, 검색·조건 단계 자체를 형제 이미지 구별에 맞춰 학습한다.

- 배치 = 형제 묶음을 이어 붙인 것 → 배치 안의 오답 대부분이 형제 이미지·형제 문장 (SigLIP sigmoid 손실)
- 문장 최소 쌍: 이미지마다 한 가지만 바꾼 문장 M개를 추가 오답으로 (NegCLIP 계열, 규칙 기반)
- 학습 대상: 이미지·문장 인코더의 마지막 n층 + 출력 헤드. 앞쪽 층은 고정하고 이미지 앞쪽 출력은 한 번만 계산해 보관한다.
- 온도·편향(logit_scale, logit_bias)은 고정해 점수 범위를 유지한다.
학습 데이터의 이미지와 문장만 쓴다.
"""

import math
import random
import time

import torch
from torch.nn.functional import logsigmoid, normalize

from haiku.minpairs import perturb
from haiku.siblings import build_groups, parse_prompt

SIGLIP_ID = "google/siglip-base-patch16-384"


def _pool(o):
    return o.pooler_output if hasattr(o, "pooler_output") else o


def _layer(lyr, h):
    try:
        o = lyr(h, None)
    except TypeError:
        o = lyr(h)
    return o[0] if isinstance(o, tuple) else o


def _open(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


@torch.no_grad()
def vision_prefix(net, proc, paths, n, device, bs=32):
    """이미지 인코더의 앞 (전체 − n)층 출력을 한 번만 계산해 CPU에 보관 (GPU에서는 fp16)."""
    vm = net.vision_model
    amp = device.type == "cuda"
    out = None
    for s in range(0, len(paths), bs):
        px = proc(images=[_open(p) for p in paths[s:s + bs]], return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            h = vm.embeddings(px)
            for lyr in vm.encoder.layers[:-n]:
                h = _layer(lyr, h)
        h = (h.half() if amp else h.float()).cpu()
        if out is None:
            out = torch.empty((len(paths),) + tuple(h.shape[1:]), dtype=h.dtype)
        out[s:s + len(h)] = h
    return out


def vision_tail(net, h, n):
    """보관한 앞쪽 출력 → 마지막 n층 → 이미지 벡터."""
    vm = net.vision_model
    for lyr in vm.encoder.layers[-n:]:
        h = _layer(lyr, h)
    h = vm.post_layernorm(h)
    return vm.head(h) if hasattr(vm, "head") and getattr(vm, "use_head", True) else h[:, 0]


def train(rows, train_ids, device, model_id=SIGLIP_ID, layers=4, epochs=5, lr=1e-5, batch=32, text_negs=2, seed=0, log=print):
    from transformers import AutoModel, AutoProcessor
    torch.manual_seed(seed)
    net = AutoModel.from_pretrained(model_id).to(device).float().eval()
    proc = AutoProcessor.from_pretrained(model_id)
    vm, tm = net.vision_model, net.text_model
    by_id = {r["id"]: r for r in rows}
    ids = list(train_ids)
    amp = device.type == "cuda"

    t0 = time.perf_counter()
    paths = [str(by_id[i]["image_path"]) for i in ids]
    feats = vision_prefix(net, proc, paths, layers, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):  # 나눠 계산해도 같은지 확인
        a = _pool(vision_tail(net, feats[:4].to(device), layers)).float()
        b = _pool(net.get_image_features(pixel_values=proc(images=[_open(p) for p in paths[:4]],
                                                           return_tensors="pt").pixel_values.to(device))).float()
    diff = float((normalize(a, dim=-1) - normalize(b, dim=-1)).abs().max())
    assert diff < 2e-2, f"이미지 인코더를 나눠 계산한 결과가 다릅니다 (최대 차이 {diff:.4f})"
    log(f"이미지 앞쪽 출력 {len(ids)}장 {time.perf_counter() - t0:.0f}초 (나눠 계산 확인: 최대 차이 {diff:.4f})", flush=True)

    net.requires_grad_(False)
    mods = [vm.encoder.layers[-layers:], vm.post_layernorm, tm.encoder.layers[-layers:], tm.final_layer_norm]
    mods += [m.head for m in (vm, tm) if hasattr(m, "head")]
    for mod in mods:
        mod.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tok = lambda t: proc(text=t, padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids.to(device)

    group_of, _ = build_groups(rows, {r["id"]: parse_prompt(r["prompt"]) for r in rows})
    members = {}
    for i in ids:
        members.setdefault(group_of[i], []).append(i)
    pos = {i: k for k, i in enumerate(ids)}
    total = epochs * math.ceil(len(ids) / batch)
    history, step = [], 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        rng = random.Random(seed * 1000 + ep)
        gs = list(members.values())
        rng.shuffle(gs)
        order = [i for g in gs for i in rng.sample(g, len(g))]            # 형제 묶음이 한 배치에 모이도록
        losses = []
        for s in range(0, len(order), batch):
            bt = order[s:s + batch]
            if len(bt) < 2:
                continue
            prompts = [by_id[i]["prompt"] for i in bt]
            negs = []
            for p in prompts if text_negs else []:
                pert = perturb(p, rng, text_negs)
                while len(pert) < text_negs:
                    x = by_id[rng.choice(ids)]["prompt"]
                    if x != p:
                        pert.append(x)
                negs += pert
            for gph in opt.param_groups:                                 # 처음 10%는 학습률을 서서히 올림
                gph["lr"] = lr * min(1.0, (step + 1) / max(1, int(0.1 * total)))
            B = len(bt)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                zi = _pool(vision_tail(net, feats[[pos[i] for i in bt]].to(device), layers))
                zt = _pool(net.get_text_features(input_ids=tok(prompts + negs)))
            zi, zt = normalize(zi.float(), dim=-1), normalize(zt.float(), dim=-1)
            scale, bias = net.logit_scale.exp().float(), net.logit_bias.float()
            logits = zt[:B] @ zi.T * scale + bias                          # (문장, 이미지)
            same = torch.tensor([[prompts[x] == prompts[y] for y in range(B)] for x in range(B)], device=device)
            loss = -logsigmoid(torch.where(same, 1.0, -1.0) * logits).sum() / B
            if text_negs:                                                 # 같은 이미지에서 한 가지만 다른 문장은 오답
                ln = (zt[B:].view(B, text_negs, -1) * zi[:, None, :]).sum(-1) * scale + bias
                loss = loss - logsigmoid(-ln).sum() / B
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        history.append(round(sum(losses) / max(len(losses), 1), 4))
        log(f"  epoch {ep + 1}/{epochs} loss {history[-1]:.4f}", flush=True)
    net.eval()
    return net, proc, {"loss": history, "steps": step, "train_sec": round(time.perf_counter() - t0, 1),
                       "trainable_params": sum(p.numel() for p in params), "layers": layers, "lr": lr, "batch": batch,
                       "text_negs": text_negs}


def save(net, proc, out_dir):
    net.half().save_pretrained(out_dir)
    proc.save_pretrained(out_dir)


def encoders(net, proc, device):
    """평가용 인코더 (haiku.combo._siglip과 같은 계산)."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = net.to(dtype).eval()

    @torch.inference_mode()
    def img_fn(paths):
        out = []
        for s in range(0, len(paths), 32):
            px = proc(images=[_open(p) for p in paths[s:s + 32]], return_tensors="pt").pixel_values
            out.append(normalize(_pool(net.get_image_features(pixel_values=px.to(device, dtype))).float(), dim=-1))
        return torch.cat(out)

    @torch.inference_mode()
    def txt_fn(t):
        out = []
        for s in range(0, len(t), 256):
            ids = proc(text=t[s:s + 256], padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids
            out.append(normalize(_pool(net.get_text_features(input_ids=ids.to(device))).float(), dim=-1))
        return torch.cat(out)
    return img_fn, txt_fn


def _score(ranks):
    return sum((r == 1) * 0.5 + (0.5 / math.log2(r + 1) if r <= 5 else 0.0) for r in ranks) / max(len(ranks), 1)


def stage_eval(rows, splits, g_ids, img_fn, txt_fn, lam=3.2, Ks=(10, 50)):
    """검색만 / 검색 + 조건 재정렬 단계의 점수 (BLIP 없이). splits: {이름: 질의 ID들}, g_ids: 후보 이미지 ID들 (질의 포함)."""
    from haiku.combo import prepare
    from haiku.rerank import PhraseBank, all_phrases, extract_conditions
    by_id = {r["id"]: r for r in rows}
    g_ids = list(g_ids)
    pos = {i: k for k, i in enumerate(g_ids)}
    prompts = {i: by_id[i]["prompt"] for i in g_ids}
    img = img_fn([str(by_id[i]["image_path"]) for i in g_ids])
    txt = txt_fn([prompts[i] for i in g_ids])
    out = {}
    for name, q_ids in splits.items():
        conds = {i: extract_conditions(prompts[i]) for i in q_ids}
        bank = PhraseBank(txt_fn)
        bank.add([p for i in q_ids for p in all_phrases(conds[i])])
        for K in Ks:
            items, _ = prepare(list(q_ids), g_ids, txt, img, pos, conds, bank, lam, K, None, None, prompts, 1)
            sr = [it["base_rank"] for it in items]
            cr = [it["idxB"].index(it["gold"]) + 1 if it["gold"] in it["idxB"] else max(it["base_rank"], K + 1) for it in items]
            n = len(items)
            out[(name, K)] = {"search": round(_score(sr), 4), "search_R@1": round(sum(r == 1 for r in sr) / n, 4),
                              "search_top10": round(sum(r <= 10 for r in sr) / n, 4),
                              "cond": round(_score(cr), 4), "cond_R@1": round(sum(r == 1 for r in cr) / n, 4),
                              "cond_top5": round(sum(r <= 5 for r in cr) / n, 4),
                              "cond_top10": round(sum(r <= 10 for r in cr) / n, 4)}
    return out
