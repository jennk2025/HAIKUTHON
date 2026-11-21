"""SigLIP 계열 임베딩 검색. 스타터 키트 model.py와 같은 방식(fp16, 정규화, 코사인 유사도)으로 계산하되
Top-5만이 아니라 전체 순위를 돌려준다. Hit@K, 순위 분포 분석에 쓴다.
"""

import torch
from PIL import Image
from torch.nn.functional import normalize


def load_siglip(model_id_or_dir, device="cuda", half=True, local_files_only=False):
    from transformers import AutoProcessor, SiglipModel
    network = SiglipModel.from_pretrained(model_id_or_dir, local_files_only=local_files_only).to(device)
    if half and network.device.type == "cuda":
        network.half()
    processor = AutoProcessor.from_pretrained(model_id_or_dir, local_files_only=local_files_only)
    return network.eval(), processor


def _features(out):
    # transformers 5.x의 get_*_features는 출력 객체를 돌려준다(스타터 키트는 .pooler_output 사용).
    return out.pooler_output if hasattr(out, "pooler_output") else out


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def embed_images(model, paths, batch_size=32):
    network, processor = model
    out = []
    for s in range(0, len(paths), batch_size):
        images = [read_image(p) for p in paths[s:s + batch_size]]
        pixels = processor(images=images, return_tensors="pt").pixel_values
        pixels = pixels.to(device=network.device, dtype=network.dtype)
        out.append(normalize(_features(network.get_image_features(pixel_values=pixels)).float(), dim=-1))
    return torch.cat(out)


@torch.inference_mode()
def embed_texts(model, prompts, batch_size=128, max_length=None):
    network, processor = model
    kwargs = {"padding": "max_length", "truncation": True, "return_tensors": "pt"}
    if max_length is not None:
        kwargs["max_length"] = max_length
    out = []
    for s in range(0, len(prompts), batch_size):
        tokens = processor(text=prompts[s:s + batch_size], **kwargs).to(network.device)
        out.append(normalize(_features(network.get_text_features(**tokens)).float(), dim=-1))
    return torch.cat(out)


@torch.inference_mode()
def ranks_of_gold(text_vecs, image_vecs, gold_index, chunk=1024):
    """각 query에서 정답 이미지의 순위(1부터). 정답과 점수가 같은 이미지는 정답보다 앞에 둔다(보수적)."""
    gold_index = torch.as_tensor(gold_index, device=text_vecs.device)
    ranks = []
    for s in range(0, len(text_vecs), chunk):
        scores = text_vecs[s:s + chunk] @ image_vecs.T
        gold_scores = scores.gather(1, gold_index[s:s + chunk, None])
        ranks.append((scores >= gold_scores).sum(1))  # 정답 자신 포함 → 그대로 순위
    return torch.cat(ranks).cpu().numpy()


def evaluate_rows(model, query_rows, gallery_rows, img_bs=32, txt_bs=128):
    """query_rows의 정답 이미지가 gallery_rows 안에 있어야 한다. (ranks, text_vecs, image_vecs)"""
    gallery_index = {r["id"]: i for i, r in enumerate(gallery_rows)}
    image_vecs = embed_images(model, [r["image_path"] for r in gallery_rows], img_bs)
    text_vecs = embed_texts(model, [r["prompt"] for r in query_rows], txt_bs)
    gold = [gallery_index[r["id"]] for r in query_rows]
    return ranks_of_gold(text_vecs, image_vecs, gold), text_vecs, image_vecs
