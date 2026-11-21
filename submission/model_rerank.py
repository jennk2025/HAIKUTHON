"""HAIKUTHON 제출 모델: SigLIP 검색 → 조건 분해 재정렬 → (선택) BLIP ITM 재정렬 → (선택) OWL-ViT 좌우·상하 점검.

1. 후보 이미지를 한 번 읽어 SigLIP으로 인코딩한다 (DataLoader worker로 디코딩 병렬화).
   BLIP을 쓰면 같은 크기로 줄인 이미지를 GPU에 보관해 BLIP도 다시 읽지 않고 쓴다.
2. 질의마다 원문으로 상위 K개 후보를 찾는다.
3. 질의 문장 하나에서 규칙으로 객체별 조건과 대조 문구를 만들고, 후보 K개의 조건 점수를 원래 점수에 더한다.
4. (models/blip이 있고 설정에서 켰을 때) 조건 재정렬 결과의 상위 k개를 BLIP ITM(교차 어텐션)으로 다시 채점해 더한다.
5. (models/owl이 있고 설정에서 켰을 때) 질의에 단일 물체 사이의 좌우·상하 관계가 있으면, 최종 상위 2장에서
   두 물체를 OWL-ViT로 검출해 상자 중심을 비교하고, 2위가 관계를 더 잘 지키면 두 장의 순서만 바꾼다.
   검출은 (후보 이미지, 물체 이름)마다 독립이라 같은 이미지를 보는 질의끼리 계산만 공유한다.
- 각 질의의 순위는 그 질의 문장과 후보 이미지만으로 계산한다. 다른 질의의 결과로 보정하지 않는다.
- 같은 문구의 임베딩은 재사용한다 (값이 문구만으로 정해지므로 다른 질의에 의존하지 않음).
- 재정렬·BLIP 사용 여부는 이미지 인코딩 직후, OWL 사용 여부는 OWL 단계 직전에 한 번만 정해
  모든 질의에 똑같이 적용한다 (시간 안전장치, 순위 보정이 아님).
- models/ 폴더의 가중치와 설정만 읽는다. 네트워크를 쓰지 않는다.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoModel, AutoProcessor

from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores

IMG_BATCH, TXT_BATCH, MAX_TEXT_LEN, BLIP_BATCH, ITM_QBATCH, OWL_BATCH = 64, 256, 64, 64, 128, 32
WORKERS = int(os.environ.get("HAIKU_WORKERS", min(4, os.cpu_count() or 1)))


def load_model(model_dir, device):
    root = Path(model_dir) / "models"
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = AutoModel.from_pretrained(root / "siglip", local_files_only=True, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(root / "siglip", local_files_only=True)
    cfg = json.loads((root / "haiku_config.json").read_text(encoding="utf-8"))
    model = {"net": net, "proc": proc, "device": device, "cfg": cfg, "blip": None}
    if (cfg.get("itm") or {}).get("use") and (root / "blip").exists():
        from transformers import BlipForImageTextRetrieval
        blip = BlipForImageTextRetrieval.from_pretrained(root / "blip", local_files_only=True, dtype=dtype).to(device).eval()
        model["blip"] = blip
        model["blip_proc"] = AutoProcessor.from_pretrained(root / "blip", local_files_only=True)
    model["owl"] = None
    if (cfg.get("owl") or {}).get("use") and (root / "owl").exists():
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        model["owl"] = OwlViTForObjectDetection.from_pretrained(root / "owl", local_files_only=True, dtype=dtype).to(device).eval()
        model["owl_proc"] = OwlViTProcessor.from_pretrained(root / "owl", local_files_only=True)
    return model


def _size(ip):
    """전처리 설정의 입력 크기 (dict / SizeDict / 숫자 모두 처리)."""
    s = ip.size
    for key in ("height", "shortest_edge"):
        v = getattr(s, key, None)
        if v is None and isinstance(s, dict):
            v = s.get(key)
        if v:
            return int(v)
    return int(s) if isinstance(s, (int, float)) else 384


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.ip = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.ip(images=im, return_tensors="pt")["pixel_values"][0]


class _Raw(torch.utils.data.Dataset):
    """같은 크기로 줄인 uint8 이미지 (SigLIP·BLIP이 함께 씀)."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


def _feats(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _norm(x_uint8, ip, size, dtype):
    x = x_uint8.float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    mean = torch.tensor(ip.image_mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


@torch.inference_mode()
def encode_images(model, paths):
    net, dev = model["net"], model["device"]
    dl = torch.utils.data.DataLoader(_Images(paths, model["proc"].image_processor), batch_size=IMG_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    return torch.cat([normalize(_feats(net.get_image_features(
        pixel_values=px.to(dev, dtype=net.dtype, non_blocking=True))).float(), dim=-1) for px in dl])


@torch.inference_mode()
def encode_images_raw(model, paths):
    """이미지를 한 번만 읽어 SigLIP 임베딩과 BLIP용 uint8 보관본을 함께 만든다."""
    net, dev = model["net"], model["device"]
    sip, bip = model["proc"].image_processor, model["blip_proc"].image_processor
    s_size, b_size = _size(sip), _size(bip)
    dl = torch.utils.data.DataLoader(_Raw(paths, b_size), batch_size=IMG_BATCH, num_workers=WORKERS,
                                     pin_memory=dev.type == "cuda")
    raw = torch.empty((len(paths), 3, b_size, b_size), dtype=torch.uint8, device=dev)
    vecs, s = [], 0
    for x in dl:
        x = x.to(dev, non_blocking=True)
        raw[s:s + len(x)] = x
        s += len(x)
        vecs.append(normalize(_feats(net.get_image_features(pixel_values=_norm(x, sip, s_size, net.dtype))).float(), dim=-1))
    return torch.cat(vecs), raw


@torch.inference_mode()
def encode_texts(model, prompts):
    net, proc, dev = model["net"], model["proc"], model["device"]
    out = []
    for s in range(0, len(prompts), TXT_BATCH):
        tok = proc(text=prompts[s:s + TXT_BATCH], padding="max_length", max_length=MAX_TEXT_LEN,
                   truncation=True, return_tensors="pt").to(dev)
        out.append(normalize(_feats(net.get_text_features(**tok)).float(), dim=-1))
    return torch.cat(out)


@torch.inference_mode()
def itm_scores(model, raw, img_idx, texts):
    """(문장, 이미지) 쌍마다 BLIP 매칭 로짓 차이. 이미지 특징은 이번 묶음에 필요한 것만 계산."""
    blip, bproc, dev = model["blip"], model["blip_proc"], model["device"]
    bip = bproc.image_processor
    uniq = sorted(set(img_idx))
    where = {j: n for n, j in enumerate(uniq)}
    feats = []
    for s in range(0, len(uniq), BLIP_BATCH):
        px = _norm(raw[uniq[s:s + BLIP_BATCH]], bip, _size(bip), next(blip.parameters()).dtype)
        feats.append(blip.vision_model(pixel_values=px).last_hidden_state)
    feats = torch.cat(feats)
    out = []
    for s in range(0, len(texts), BLIP_BATCH):
        tok = bproc.tokenizer(texts[s:s + BLIP_BATCH], padding=True, truncation=True, max_length=72,
                              return_tensors="pt").to(dev)
        img = feats[[where[j] for j in img_idx[s:s + BLIP_BATCH]]]
        att = torch.ones(img.shape[:-1], dtype=torch.long, device=dev)
        h = blip.text_encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                              encoder_hidden_states=img, encoder_attention_mask=att).last_hidden_state
        logit = blip.itm_head(h[:, 0, :]).float()
        out.append(logit[:, 1] - logit[:, 0])
    return torch.cat(out)


@torch.inference_mode()
def owl_detect(model, paths, need):
    """need: {후보 번호: {물체 이름}} → {(후보 번호, 이름): (검출 점수, cx, cy)}.
    이름마다 점수가 가장 높은 상자 하나 (haiku.owl.OwlChecker.detect와 같은 계산, 이미지는 한 번만 읽음)."""
    owl, proc, dev = model["owl"], model["owl_proc"], model["device"]
    names = sorted({n for v in need.values() for n in v})
    row = {n: k for k, n in enumerate(names)}
    tok = proc.tokenizer([f"a photo of a {n}" for n in names] + [" "], padding="max_length", truncation=True,
                         return_tensors="pt").to(dev)
    pad = len(names)
    js = sorted(need)
    dl = torch.utils.data.DataLoader(_Images([paths[j] for j in js], proc.image_processor), batch_size=OWL_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    out, s = {}, 0
    for px in dl:
        chunk = js[s:s + len(px)]
        s += len(px)
        lists = [sorted(need[j]) for j in chunk]
        width = max(len(l) for l in lists)
        ridx = torch.tensor([[row[n] for n in l] + [pad] * (width - len(l)) for l in lists], device=dev).view(-1)
        o = owl(input_ids=tok["input_ids"][ridx], attention_mask=tok["attention_mask"][ridx],
                pixel_values=px.to(dev, dtype=next(owl.parameters()).dtype, non_blocking=True))
        probs, boxes = o.logits.float().sigmoid(), o.pred_boxes.float()
        best = probs.argmax(dim=1)                                          # (B, 질의 수)
        for b, (j, l) in enumerate(zip(chunk, lists)):
            for q, n in enumerate(l):
                p = int(best[b, q])
                out[(j, n)] = (float(probs[b, p, q]), float(boxes[b, p, 0]), float(boxes[b, p, 1]))
    return out


def owl_rerank(model, queries, paths, orders, ocfg, t0, t_img_per):
    """관계 질의의 최종 상위 2장만 OWL 규칙으로 다시 본다. 시간이 모자랄 것 같으면 모든 질의에서 건너뛴다."""
    from haiku.owl import relation_score, spatial_relations
    rels, need = {}, {}
    for n, q in enumerate(queries):
        r = spatial_relations(q["prompt"])
        if r and len(orders[n]) >= 2:
            names = tuple(sorted({x for _, a, b in r for x in (a, b)}))
            rels[n] = (r, names)
            for j in orders[n][:2]:
                need.setdefault(j, set()).update(names)
    info = {"owl_queries": len(rels), "owl_images": len(need), "owl_used": False}
    elapsed = time.perf_counter() - t0
    projected = len(need) * t_img_per * float(ocfg.get("cost_factor", 1.5))
    info.update(owl_projected_sec=round(projected, 1), owl_start_sec=round(elapsed, 1))
    if not rels or elapsed + projected > float(ocfg.get("budget_sec", 830)):
        return info
    det = owl_detect(model, paths, need)
    thr, margin = float(ocfg.get("thr", 0.1)), float(ocfg.get("margin", 0.06))
    swapped = 0
    for n, (r, names) in rels.items():
        a, b = orders[n][:2]
        sa = relation_score({x: det[(a, x)] for x in names}, r, thr, margin)
        sb = relation_score({x: det[(b, x)] for x in names}, r, thr, margin)
        if sb > sa:
            orders[n] = [b, a] + orders[n][2:]
            swapped += 1
    info.update(owl_used=True, owl_swapped=swapped, owl_sec=round(time.perf_counter() - t0 - elapsed, 1))
    return info


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    t0 = time.perf_counter()
    cfg = model["cfg"]
    K, method, lam = int(cfg.get("K", 10)), cfg.get("method", "B"), float(cfg.get("lam", 0.0))
    reliability, tau = cfg.get("reliability", {}), cfg.get("tau")
    kinds = set(cfg["kinds"]) if cfg.get("kinds") else None          # 쓸 조건 종류 (없으면 전부)
    icfg = cfg.get("itm") or {}
    image_ids = [c["image_id"] for c in candidates]
    paths = [c["image_path"] for c in candidates]
    if model.get("blip") is not None:
        image_vecs, raw = encode_images_raw(model, paths)
    else:
        image_vecs, raw = encode_images(model, paths), None
    t_img = time.perf_counter() - t0
    timing = {"encode_sec": round(t_img, 1)}
    # 재정렬·BLIP 사용 여부는 여기서 한 번만 정한다 (질의 내용과 무관, 모든 질의에 동일)
    elapsed = time.perf_counter() - t0
    use_rerank = method != "A" and lam > 0 and elapsed < float(cfg.get("skip_rerank_after_sec", 700))
    use_itm = raw is not None and icfg.get("use") and elapsed < float(icfg.get("skip_after_sec", 600))
    k_itm, lam_itm, tau_itm = int(icfg.get("k", 2)), float(icfg.get("lam", 0.0)), icfg.get("tau")
    bank = PhraseBank(lambda t: encode_texts(model, t))
    orders, heads = [], []
    for s in range(0, len(queries), TXT_BATCH):
        batch = queries[s:s + TXT_BATCH]
        text_vecs = encode_texts(model, [q["prompt"] for q in batch])
        top = (text_vecs @ image_vecs.T).topk(max(K, top_k), dim=1)
        conds = [[c for c in extract_conditions(q["prompt"]) if kinds is None or c["kind"] in kinds]
                 for q in batch] if use_rerank else None
        if use_rerank:
            bank.add([p for c in conds for p in all_phrases(c)])
        for i in range(len(batch)):
            idx, s_k = top.indices[i], top.values[i]
            new = s_k[:K]
            if use_rerank and conds[i]:
                M = condition_margins(conds[i], bank, image_vecs[idx[:K]])
                new = rerank_scores(s_k[:K], conds[i], M, method, lam, reliability, tau)
            o = new.argsort(descending=True)
            orders.append(torch.cat([idx[:K][o], idx[K:]]).tolist())
            heads.append(new[o].float())
    timing["rerank_end_sec"] = round(time.perf_counter() - t0, 1)
    if use_itm and lam_itm > 0:
        k_itm = min(k_itm, K)
        sel = [n for n in range(len(queries)) if tau_itm is None or float(heads[n][0] - heads[n][1]) < tau_itm]
        # 같은 후보를 보는 질의끼리 묶어 BLIP 이미지 특징 재계산을 줄인다 (각 질의의 결과는 묶음과 무관)
        sel.sort(key=lambda n: orders[n][0])
        for s in range(0, len(sel), ITM_QBATCH):
            chunk = sel[s:s + ITM_QBATCH]
            texts = [queries[n]["prompt"] for n in chunk for _ in range(k_itm)]
            img_idx = [orders[n][j] for n in chunk for j in range(k_itm)]
            sc = itm_scores(model, raw, img_idx, texts).view(len(chunk), k_itm)
            for r, n in enumerate(chunk):
                h = heads[n][:k_itm].to(sc) + lam_itm * sc[r]
                head = [orders[n][j] for j in h.argsort(descending=True).tolist()]
                orders[n] = head + orders[n][k_itm:]
    timing["itm_used"] = bool(use_itm and lam_itm > 0)
    timing["itm_end_sec"] = round(time.perf_counter() - t0, 1)
    ocfg = cfg.get("owl") or {}
    if model.get("owl") is not None and ocfg.get("use"):
        timing.update(owl_rerank(model, queries, paths, orders, ocfg, t0, t_img / max(len(paths), 1)))
    timing["total_sec"] = round(time.perf_counter() - t0, 1)
    model["last_timing"] = timing
    return {q["prompt_id"]: [image_ids[j] for j in orders[n][:top_k]] for n, q in enumerate(queries)}
