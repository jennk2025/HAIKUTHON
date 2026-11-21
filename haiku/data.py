"""Train 데이터 읽기, 공식 분할 재현, 스타터 키트 predict() 입력 형식 변환."""

import json
import random
from pathlib import Path


def load_rows(data_dir):
    """[{"id": "0000", "prompt": "...", "image_path": Path}, ...] (숫자 ID 오름차순)"""
    data_dir = Path(data_dir)
    with (data_dir / "prompts.jsonl").open(encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    rows = [
        {"id": r["id"], "prompt": r["prompts"], "image_path": data_dir / "images" / f"{r['id']}.png"}
        for r in raw
    ]
    rows.sort(key=lambda r: int(r["id"]))
    return rows


def official_split(rows, seed=42):
    """스타터 키트 train.py의 load_and_split과 같은 분할 (학습 80% / 검증 20%)."""
    ids = [r["id"] for r in rows]
    n_train = len(rows) * 8 // 10
    train_ids = set(random.Random(seed).sample(ids, n_train))
    train = [r for r in rows if r["id"] in train_ids]
    valid = [r for r in rows if r["id"] not in train_ids]
    return train, valid


def load_split(split_path, rows):
    """splits/split_seed42.json을 읽어 (train_rows, valid_rows)를 돌려준다. ID 집합이 데이터와 다르면 중단한다."""
    s = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train, valid = set(s["train_ids"]), set(s["valid_ids"])
    if train & valid:
        raise ValueError("분할 파일에서 학습·검증 ID가 겹칩니다.")
    ids = {r["id"] for r in rows}
    if ids != train | valid:
        raise ValueError(f"분할 파일과 데이터의 ID가 다릅니다 (데이터 {len(ids)}, 분할 {len(train | valid)}).")
    return [r for r in rows if r["id"] in train], [r for r in rows if r["id"] in valid]


def assert_not_valid(ids, split_path):
    """학습·증강에 쓰려는 ID에 검증 ID가 섞이지 않았는지 확인한다."""
    valid = set(json.loads(Path(split_path).read_text(encoding="utf-8"))["valid_ids"])
    leaked = sorted(set(ids) & valid)
    if leaked:
        raise ValueError(f"검증 ID {len(leaked)}개가 섞여 있습니다 (예: {leaked[:5]}).")


def to_queries(rows):
    return [{"prompt_id": r["id"], "prompt": r["prompt"]} for r in rows]


def to_candidates(rows):
    return [{"image_id": r["id"], "image_path": r["image_path"]} for r in rows]


def gold_of(rows):
    """Train에서는 프롬프트 ID와 정답 이미지 ID가 같다."""
    return {r["id"]: r["id"] for r in rows}
