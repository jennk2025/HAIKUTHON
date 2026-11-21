"""HAIKUTHON 제출 모델: SigLIP 검색 + 조건 분해 재정렬.

1. 후보 이미지를 SigLIP으로 한 번 인코딩한다 (DataLoader worker로 디코딩 병렬화).
2. 질의마다 원문으로 상위 K개 후보를 찾는다.
3. 질의 문장 하나에서 규칙으로 객체별 조건(색·개수·상태·결합·관계)과 대조 문구를 만들고,
   후보 K개에 대한 조건 점수를 원래 점수에 더해 순위를 조정한다 (후보를 탈락시키지 않음).
- 각 질의의 순위는 그 질의 문장과 후보 이미지만으로 계산한다. 다른 질의의 결과로 보정하지 않는다.
- 같은 문구의 임베딩은 재사용한다 (값이 문구만으로 정해지므로 다른 질의에 의존하지 않음).
- 재정렬을 할지는 이미지 인코딩 직후 한 번만 정해 모든 질의에 똑같이 적용한다 (시간 안전장치).
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

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoModel, AutoProcessor

from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores

IMG_BATCH, TXT_BATCH, MAX_TEXT_LEN = 64, 256, 64
WORKERS = int(os.environ.get("HAIKU_WORKERS", min(4, os.cpu_count() or 1)))


def load_model(model_dir, device):
    root = Path(model_dir) / "models"
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = AutoModel.from_pretrained(root / "siglip", local_files_only=True, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(root / "siglip", local_files_only=True)
    cfg = json.loads((root / "haiku_config.json").read_text(encoding="utf-8"))
    return {"net": net, "proc": proc, "device": device, "cfg": cfg}


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.ip = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.ip(images=im, return_tensors="pt")["pixel_values"][0]


def _feats(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


@torch.inference_mode()
def encode_images(model, paths):
    net, dev = model["net"], model["device"]
    dl = torch.utils.data.DataLoader(_Images(paths, model["proc"].image_processor), batch_size=IMG_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    return torch.cat([normalize(_feats(net.get_image_features(
        pixel_values=px.to(dev, dtype=net.dtype, non_blocking=True))).float(), dim=-1) for px in dl])


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
def predict(model, queries, candidates, top_k=5):
    t0 = time.perf_counter()
    cfg = model["cfg"]
    K, method, lam = int(cfg.get("K", 10)), cfg.get("method", "B"), float(cfg.get("lam", 0.0))
    reliability, tau = cfg.get("reliability", {}), cfg.get("tau")
    kinds = set(cfg["kinds"]) if cfg.get("kinds") else None          # 쓸 조건 종류 (없으면 전부)
    image_ids = [c["image_id"] for c in candidates]
    image_vecs = encode_images(model, [c["image_path"] for c in candidates])
    # 재정렬 여부는 여기서 한 번만 정한다 (질의 내용과 무관, 모든 질의에 동일)
    use_rerank = method != "A" and lam > 0 and (time.perf_counter() - t0) < float(cfg.get("skip_rerank_after_sec", 700))
    bank = PhraseBank(lambda t: encode_texts(model, t))
    predictions = {}
    for s in range(0, len(queries), TXT_BATCH):
        batch = queries[s:s + TXT_BATCH]
        text_vecs = encode_texts(model, [q["prompt"] for q in batch])
        top = (text_vecs @ image_vecs.T).topk(max(K, top_k), dim=1)
        conds = [[c for c in extract_conditions(q["prompt"]) if kinds is None or c["kind"] in kinds]
                 for q in batch] if use_rerank else None
        if use_rerank:
            bank.add([p for c in conds for p in all_phrases(c)])
        for i, q in enumerate(batch):
            idx, s_k = top.indices[i], top.values[i]
            if use_rerank and conds[i]:
                M = condition_margins(conds[i], bank, image_vecs[idx[:K]])
                new = rerank_scores(s_k[:K], conds[i], M, method, lam, reliability, tau)
                order = torch.cat([idx[:K][new.argsort(descending=True)], idx[K:]])
            else:
                order = idx
            predictions[q["prompt_id"]] = [image_ids[j] for j in order[:top_k].tolist()]
    return predictions
