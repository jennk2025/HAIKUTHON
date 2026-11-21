"""시간·GPU 메모리 측정과 실험 결과 저장."""

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch


@contextmanager
def timed(record, key):
    """with timed(rec, "encode"): ... → rec["encode"] = 초. GPU 비동기 실행을 고려해 동기화한다."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record[key] = round(time.perf_counter() - t0, 2)


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def test_scale_inputs(rows, n_images=16000, n_queries=8000):
    """Test 규모(이미지 16,000 / query 8,000)의 predict() 입력을 Train을 반복해서 만든다.
    같은 파일을 여러 번 읽으므로 두 번째부터는 OS 파일 캐시 덕분에 실제보다 빠를 수 있다.
    """
    candidates = [
        {"image_id": f"{k}_{r['id']}", "image_path": r["image_path"]}
        for k in range(-(-n_images // len(rows)))
        for r in rows
    ][:n_images]
    queries = [
        {"prompt_id": f"{k}_{r['id']}", "prompt": r["prompt"]}
        for k in range(-(-n_queries // len(rows)))
        for r in rows
    ][:n_queries]
    return queries, candidates


def save_result(results_dir, exp_id, payload):
    """results_dir/<exp_id>.json 저장 (실행 시각 포함)."""
    payload = {"exp_id": exp_id, "saved_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path = Path(results_dir) / f"{exp_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"저장: {path}")
    return path
