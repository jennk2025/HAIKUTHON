"""OWL로 찾은 물체 영역 안에서 색 비교: '회색인 게 스카프인가, 가방인가?'를 물체 영역으로 좁혀 본다.

- 조건: 규칙 파서의 color_x (물체 하나에 색 하나, 대조 = 같은 질의의 다른 물체 색). 여러 개인 물체·여러 색 물체는 파서에서 이미 제외.
- 후보 이미지마다 OWL-ViT로 그 물체의 상자를 찾고(검출 점수 ≥ thr), 상자를 조금 넓혀 잘라 SigLIP으로 인코딩한다.
  여백 = target 문구 점수 − 대조 문구 중 최고 점수 ("a gray scarf" vs "an orange scarf").
- 두 후보를 비교할 때는 두 후보 모두에서 물체를 찾은 조건만 쓴다 (한쪽만 찾으면 판단 보류).
- 각 질의의 점수는 그 질의 문장과 그 후보 이미지만으로 계산한다. 다른 질의·후보의 원래 설명은 쓰지 않는다.
"""

import time

import torch
from PIL import Image
from torch.nn.functional import normalize

from haiku.rerank import extract_conditions

SIGLIP_ID = "google/siglip-base-patch16-384"


def crop_conditions(prompt):
    """질의 하나 → [(물체 이름, target 문구, [대조 문구])]."""
    out = []
    for c in extract_conditions(prompt):
        if c["kind"] == "color_x" and c["contrasts"]:
            out.append((c["key"].split("[", 1)[1].split("]", 1)[0], c["target"], list(c["contrasts"])))
    return out


class CropColor:
    def __init__(self, owl_chk, siglip_id=SIGLIP_ID, device="cuda", pad=0.1, thr=0.1, bs=64, local_files_only=False):
        from transformers import AutoModel, AutoProcessor
        self.owl, self.device = owl_chk, torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.net = AutoModel.from_pretrained(siglip_id, dtype=dtype, local_files_only=local_files_only).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(siglip_id, local_files_only=local_files_only)
        self.dtype, self.pad, self.thr, self.bs = dtype, pad, thr, bs
        self.txt, self.crop, self.full = {}, {}, {}
        self.n_crops, self.crop_sec = 0, 0.0

    @staticmethod
    def _f(o):
        return o.pooler_output if hasattr(o, "pooler_output") else o

    @torch.inference_mode()
    def _enc_text(self, phrases):
        todo = [p for p in dict.fromkeys(phrases) if p not in self.txt]
        for s in range(0, len(todo), 256):
            chunk = todo[s:s + 256]
            tok = self.proc(text=chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
            for p, v in zip(chunk, normalize(self._f(self.net.get_text_features(**tok)).float(), dim=-1).cpu()):
                self.txt[p] = v

    @torch.inference_mode()
    def _enc_images(self, keys, make):
        """keys: 캐시 키 목록, make(key) → PIL 이미지. 결과는 정규화 벡터."""
        out = {}
        for s in range(0, len(keys), self.bs):
            chunk = keys[s:s + self.bs]
            px = self.proc(images=[make(k) for k in chunk], return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            for k, v in zip(chunk, normalize(self._f(self.net.get_image_features(pixel_values=px)).float(), dim=-1).cpu()):
                out[k] = v
        return out

    def _crop_image(self, path, box):
        _, cx, cy, w, h = box
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x0, x1 = (cx - w / 2 - self.pad * w) * W, (cx + w / 2 + self.pad * w) * W
            y0, y1 = (cy - h / 2 - self.pad * h) * H, (cy + h / 2 + self.pad * h) * H
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, max(int(x1), x0 + 8)), min(H, max(int(y1), y0 + 8))
            return im.crop((x0, y0, x1, y1))

    def margins(self, items):
        """items: [(image_path, prompt)] → 항목마다 [(이름, 잘라 본 여백 또는 None, 전체 이미지 여백)]."""
        conds = {p: crop_conditions(p) for p in dict.fromkeys(p for _, p in items)}
        names = {p: tuple(sorted({n for n, _, _ in cs})) for p, cs in conds.items()}
        self.owl.detect([(path, names[p]) for path, p in items if conds[p]])
        self._enc_text([t for cs in conds.values() for _, a, b in cs for t in [a] + b])
        todo_crop, todo_full = {}, {}
        for path, p in items:
            if not conds[p]:
                continue
            det = self.owl.cache[(path, names[p])]
            if path not in self.full:
                todo_full[path] = None
            for n, _, _ in conds[p]:
                if det[n][0] >= self.thr and (path, n) not in self.crop:
                    todo_crop[(path, n)] = det[n]
        t0 = time.perf_counter()
        self.crop.update(self._enc_images(list(todo_crop), lambda k: self._crop_image(k[0], todo_crop[k])))
        self.crop_sec += time.perf_counter() - t0
        self.n_crops += len(todo_crop)

        def whole(path):
            with Image.open(path) as im:
                return im.convert("RGB")
        self.full.update(self._enc_images(list(todo_full), whole))
        out = []
        for path, p in items:
            row = []
            for n, tgt, cons in conds[p]:
                T, C = self.txt[tgt], torch.stack([self.txt[c] for c in cons])
                fv = self.full[path]
                m_full = float(fv @ T - (C @ fv).max())
                cv = self.crop.get((path, n))
                row.append((n, None if cv is None else float(cv @ T - (C @ cv).max()), m_full))
            out.append(row)
        return out


def pair_scores(ma, mb, mode="crop"):
    """같은 질의의 두 후보 a, b의 조건별 여백 → (a 점수, b 점수).
    crop: 두 후보 모두에서 물체를 찾은 조건만 / whole: 전체 이미지 / hybrid: 둘 다 찾으면 잘라 보고 아니면 전체."""
    sa = sb = 0.0
    used = 0
    for (_, ca, fa), (_, cb, fb) in zip(ma, mb):
        both = ca is not None and cb is not None
        if mode == "whole" or (mode == "hybrid" and not both):
            sa, sb = sa + fa, sb + fb
        elif both:
            sa, sb, used = sa + ca, sb + cb, used + 1
    return sa, sb, used


def probe_swap_color(cc, rows, train_ids, pairs_csv, contrast_csv, per_cat=200):
    """색 교환 형제 쌍(학습 분할 안)에서 각 설명이 자기 이미지를 더 높게 보는지: 전체 이미지 vs 물체 영역."""
    from haiku.itm import load_probe_pairs
    by_id = {r["id"]: r for r in rows}
    orig, _ = load_probe_pairs(pairs_csv, contrast_csv, train_ids, per_cat)
    pairs = [x for x in orig if x[0] == "swap:color"]
    items = [(by_id[i]["image_path"], t) for _, ta, tb, ia, ib in pairs for t in (ta, tb) for i in (ia, ib)]
    t0 = time.perf_counter()
    m = cc.margins(items)
    res = {"n_pairs": len(pairs), "sec": round(time.perf_counter() - t0, 1)}
    for mode in ("whole", "crop", "hybrid"):
        both = decided = 0
        for k in range(len(pairs)):
            a_aa, a_ab, b_ba, b_bb = m[4 * k:4 * k + 4]          # (설명a,이미지a) (설명a,이미지b) (설명b,이미지a) (설명b,이미지b)
            s1, s2, u1 = pair_scores(a_aa, a_ab, mode)
            s4, s3, u2 = pair_scores(b_bb, b_ba, mode)
            if mode != "crop" or (u1 and u2):
                decided += 1
            both += (s1 > s2) and (s4 > s3)
        res[mode] = {"both_correct": both, "decided": decided, "ratio": f"{both}/{len(pairs)}"}
    return res
