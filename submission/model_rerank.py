"""HAIKUTHON 제출 모델: SigLIP 검색 → 조건 분해 재정렬 → (선택) BLIP ITM 재정렬 → (선택) OWL-ViT 좌우·상하 점검.

1. 후보 이미지를 한 번 읽어 SigLIP으로 인코딩한다 (DataLoader worker로 디코딩 병렬화).
   BLIP을 쓰면 같은 크기로 줄인 이미지를 GPU에 보관해 BLIP도 다시 읽지 않고 쓴다.
2. 질의마다 원문으로 상위 K개 후보를 찾는다.
3. 질의 문장 하나에서 규칙으로 객체별 조건과 대조 문구를 만들고, 후보 K개의 조건 점수를 원래 점수에 더한다.
4. (models/blip이 있고 설정에서 켰을 때) 조건 재정렬 결과의 상위 k개를 BLIP ITM(교차 어텐션)으로 다시 채점해 더한다.
5. (설정 color.use) 물체별 색 비교: 질의의 color_x 조건(물체 하나에 색 하나, 대조 = 같은 질의의 다른 물체 색)마다
   상위 2장에서 OWL로 그 물체 상자를 찾아 잘라 SigLIP으로 색 문구를 비교한다. 두 후보 모두에서 찾은 조건만 쓰고,
   2위의 (점수 + μ × 색 여백)이 더 크면 두 장의 순서만 바꾼다.
6. (설정 owl.use) 질의에 단일 물체 사이의 관계(owl.kinds: 좌우·상하, 확장 시 위에 놓임·안·타고 있음·옆)가 있으면
   최종 상위 2장에서 두 물체를 OWL-ViT로 검출해 상자 규칙으로 비교하고, 2위가 관계를 더 잘 지키면 순서만 바꾼다.
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
    if ((cfg.get("owl") or {}).get("use") or (cfg.get("color") or {}).get("use")) and (root / "owl").exists():
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
def owl_detect(model, paths, need, crop_need=None, crop_thr=0.1, pad=0.1):
    """need: {후보 번호: {물체 이름}} → det {(후보 번호, 이름): (검출 점수, cx, cy, w, h)} (이름마다 점수가 가장 높은 상자 하나,
    haiku.owl.OwlChecker.detect와 같은 계산, 이미지는 한 번만 읽음).
    crop_need: {후보 번호: {이름}} — 검출 점수가 crop_thr 이상인 상자를 10% 넓혀 잘라 SigLIP 벡터를 만든다 (같은 읽기 재사용).
    반환: (det, crops {(후보 번호, 이름): 정규화 벡터})"""
    owl, proc, dev = model["owl"], model["owl_proc"], model["device"]
    names = sorted({n for v in need.values() for n in v})
    row = {n: k for k, n in enumerate(names)}
    tok = proc.tokenizer([f"a photo of a {n}" for n in names] + [" "], padding="max_length", truncation=True,
                         return_tensors="pt").to(dev)
    pad_row = len(names)
    js = sorted(need)
    ip = proc.image_processor
    dl = torch.utils.data.DataLoader(_Images([paths[j] for j in js], ip), batch_size=OWL_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    o_mean = torch.tensor(ip.image_mean, device=dev).view(3, 1, 1)
    o_std = torch.tensor(ip.image_std, device=dev).view(3, 1, 1)
    net, sip = model["net"], model["proc"].image_processor
    s_size = _size(sip)
    s_mean = torch.tensor(sip.image_mean, device=dev).view(1, 3, 1, 1)
    s_std = torch.tensor(sip.image_std, device=dev).view(1, 3, 1, 1)
    det, crops, pending = {}, {}, []

    def flush():
        if pending:
            x = ((torch.stack([t for _, t in pending]) - s_mean) / s_std).to(net.dtype)
            v = normalize(_feats(net.get_image_features(pixel_values=x)).float(), dim=-1)
            for (key, _), vec in zip(pending, v):
                crops[key] = vec
            pending.clear()

    s = 0
    for px in dl:
        chunk = js[s:s + len(px)]
        s += len(px)
        lists = [sorted(need[j]) for j in chunk]
        width = max(len(l) for l in lists)
        ridx = torch.tensor([[row[n] for n in l] + [pad_row] * (width - len(l)) for l in lists], device=dev).view(-1)
        pxd = px.to(dev, non_blocking=True)
        o = owl(input_ids=tok["input_ids"][ridx], attention_mask=tok["attention_mask"][ridx],
                pixel_values=pxd.to(next(owl.parameters()).dtype))
        probs, boxes = o.logits.float().sigmoid(), o.pred_boxes.float()
        best = probs.argmax(dim=1)                                          # (B, 질의 수)
        for b, (j, l) in enumerate(zip(chunk, lists)):
            for q, n in enumerate(l):
                p = int(best[b, q])
                box = (float(probs[b, p, q]),) + tuple(float(x) for x in boxes[b, p, :4])
                det[(j, n)] = box
                if crop_need and n in crop_need.get(j, ()) and box[0] >= crop_thr:
                    img = (pxd[b].float() * o_std + o_mean).clamp(0, 1)          # OWL 입력을 원래 화소로 되돌림
                    H, W = img.shape[1:]
                    _, cx, cy, w, h = box
                    x0 = min(max(0, int((cx - w / 2 - pad * w) * W)), W - 8)
                    y0 = min(max(0, int((cy - h / 2 - pad * h) * H)), H - 8)
                    x1 = min(W, max(int((cx + w / 2 + pad * w) * W), x0 + 8))
                    y1 = min(H, max(int((cy + h / 2 + pad * h) * H), y0 + 8))
                    c = F.interpolate(img[:, y0:y1, x0:x1][None], size=(s_size, s_size), mode="bicubic",
                                      antialias=True, align_corners=False)[0].clamp(0, 1)
                    pending.append(((j, n), c))
                    if len(pending) >= IMG_BATCH:
                        flush()
    flush()
    return det, crops


def _crop_conds(prompt):
    """질의 하나 → [(물체 이름, target 문구, [대조 문구])] (color_x: 물체 하나에 색 하나, 대조 = 다른 물체의 색)."""
    out = []
    for c in extract_conditions(prompt):
        if c["kind"] == "color_x" and c["contrasts"]:
            out.append((c["key"].split("[", 1)[1].split("]", 1)[0], c["target"], list(c["contrasts"])))
    return out


def owl_rerank(model, queries, paths, orders, scores, ocfg, ccfg, t0, t_img_per):
    """최종 상위 2장만 다시 본다: (켜져 있으면) 물체별 색 → OWL 관계 규칙 순서.
    시간 안전장치는 이 단계 직전에 한 번만, 모든 질의에 똑같이 정한다: 둘 다 → OWL 관계만 → 둘 다 건너뜀."""
    from haiku.owl import relation_score, spatial_relations
    kinds = tuple(ocfg.get("kinds") or ("left_of", "above"))
    rels, need = {}, {}
    if ocfg.get("use"):
        for n, q in enumerate(queries):
            r = spatial_relations(q["prompt"], kinds)
            if r and len(orders[n]) >= 2:
                names = tuple(sorted({x for _, a, b in r for x in (a, b)}))
                rels[n] = (r, names)
                for j in orders[n][:2]:
                    need.setdefault(j, set()).update(names)
    cols, cneed = {}, {}
    if ccfg.get("use"):
        for n, q in enumerate(queries):
            cs = _crop_conds(q["prompt"])
            if cs and len(orders[n]) >= 2:
                cols[n] = cs
                for j in orders[n][:2]:
                    cneed.setdefault(j, set()).update(x for x, _, _ in cs)
    all_need = {j: set(v) for j, v in need.items()}
    for j, v in cneed.items():
        all_need.setdefault(j, set()).update(v)
    per = t_img_per * float(ocfg.get("cost_factor", 1.5))
    budget = float(ocfg.get("budget_sec", 830))
    elapsed = time.perf_counter() - t0
    info = {"owl_queries": len(rels), "color_queries": len(cols), "owl_images": len(all_need), "owl_rel_images": len(need), "owl_used": False,
            "color_used": False, "owl_start_sec": round(elapsed, 1), "owl_projected_sec": round(len(all_need) * per, 1)}
    if "skip_after_sec" in ocfg:
        # 경과 시간만으로 정한다 (다른 질의의 예측에서 나온 값은 쓰지 않음). 기준은 서버 규모 시간 측정으로 정한 상수.
        both_ok = elapsed < float(ccfg.get("skip_after_sec", ocfg["skip_after_sec"]))
        owl_ok = elapsed < float(ocfg["skip_after_sec"])
    else:                                                                   # v4 방식 (예상 시간 계산)
        both_ok = elapsed + len(all_need) * per <= budget
        owl_ok = elapsed + len(need) * per <= budget
    if cols and both_ok:
        pass                                                                # 색 + 관계 모두
    elif rels and owl_ok:
        cols, cneed, all_need = {}, {}, need                                # 시간이 빠듯하면 관계만
    else:
        return info
    det, crops = owl_detect(model, paths, all_need, cneed or None, float(ccfg.get("thr", 0.1)), float(ccfg.get("pad", 0.1)))
    if cols:
        phrases = sorted({t for cs in cols.values() for _, a, b in cs for t in [a] + b})
        tv = dict(zip(phrases, encode_texts(model, phrases)))
        mu, swapped = float(ccfg.get("mu", 0.0)), 0
        for n, cs in cols.items():
            a, b = orders[n][:2]
            sa = sb = 0.0
            for name, tgt, cons in cs:
                va, vb = crops.get((a, name)), crops.get((b, name))
                if va is None or vb is None:                                # 두 후보 모두에서 찾은 조건만
                    continue
                T, C = tv[tgt], torch.stack([tv[c] for c in cons])
                sa += float(va @ T - (C @ va).max())
                sb += float(vb @ T - (C @ vb).max())
            if scores[n][1] + mu * sb > scores[n][0] + mu * sa:
                orders[n] = [b, a] + orders[n][2:]
                scores[n] = [scores[n][1], scores[n][0]] + scores[n][2:]
                swapped += 1
        info.update(color_used=True, color_swapped=swapped, crops=len(crops))
    if rels:
        thr, margin = float(ocfg.get("thr", 0.1)), float(ocfg.get("margin", 0.06))
        swapped = 0
        for n, (r, names) in rels.items():
            a, b = orders[n][:2]
            sa = relation_score({x: det[(a, x)] for x in names}, r, thr, margin)
            sb = relation_score({x: det[(b, x)] for x in names}, r, thr, margin)
            if sb > sa:
                orders[n] = [b, a] + orders[n][2:]
                scores[n] = [scores[n][1], scores[n][0]] + scores[n][2:]
                swapped += 1
        info.update(owl_used=True, owl_swapped=swapped)
    info["owl_sec"] = round(time.perf_counter() - t0 - elapsed, 1)
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
                hs, hi = h.sort(descending=True)
                orders[n] = [orders[n][j] for j in hi.tolist()] + orders[n][k_itm:]
                heads[n] = torch.cat([hs, heads[n][k_itm:].to(hs)])
    timing["itm_used"] = bool(use_itm and lam_itm > 0)
    timing["itm_end_sec"] = round(time.perf_counter() - t0, 1)
    ocfg, ccfg = cfg.get("owl") or {}, cfg.get("color") or {}
    if model.get("owl") is not None and (ocfg.get("use") or ccfg.get("use")):
        scores = [heads[n].float().tolist() for n in range(len(queries))]
        timing.update(owl_rerank(model, queries, paths, orders, scores, ocfg, ccfg, t0, t_img / max(len(paths), 1)))
    timing["total_sec"] = round(time.perf_counter() - t0, 1)
    model["last_timing"] = timing
    return {q["prompt_id"]: [image_ids[j] for j in orders[n][:top_k]] for n, q in enumerate(queries)}
