"""Kaggle 세션 시작 도우미: 입력 데이터셋에서 학습 데이터·v3·v4 제출 폴더를 찾는다.

- Kaggle이 ZIP을 풀어 뒀으면 그 위치를 쓰고, ZIP 그대로면 /kaggle/working/work 아래에 한 번만 푼다.
- 제출 폴더는 models/haiku_config.json 으로 찾고, models/owl 이 있으면 v4, 없으면 v3로 본다.
- 파일 이름이 sub_v4.zip 이든 v4.zip 이든 찾는다.
"""

import zipfile
from pathlib import Path

INP = Path("/kaggle/input")
WORK = Path("/kaggle/working/work")


def _unzip(z, out):
    print("압축 풀기:", z.name, "→", out, flush=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)


def find_data(inp=INP, work=WORK):
    out = work / "haikuthon-train"
    hits = sorted(inp.rglob("prompts.jsonl")) or (sorted(out.rglob("prompts.jsonl")) if out.exists() else [])
    if not hits:
        zips = [z for z in sorted(inp.rglob("*.zip")) if "train" in z.name.lower()]
        if not zips:
            return None
        _unzip(zips[0], out)
        hits = sorted(out.rglob("prompts.jsonl"))
    return hits[0].parent


def find_sub(version, inp=INP, work=WORK):
    """version: 'v3' 또는 'v4'. models/owl 이 있으면 v4, 없으면 v3."""
    out = work / f"sub_{version}"
    want_owl = version == "v4"
    for r in [inp] + ([out] if out.exists() else []):
        for cfg in sorted(r.rglob("haiku_config.json")):
            d = cfg.parent.parent
            if (d / "models" / "owl").is_dir() == want_owl:
                return d
    zips = [z for z in sorted(inp.rglob("*.zip")) if version in z.stem.lower() and "train" not in z.name.lower()]
    if not zips:
        return None
    _unzip(zips[0], out)
    for cfg in sorted(out.rglob("haiku_config.json")):
        return cfg.parent.parent
    return None


def locate(inp=INP, work=WORK, verbose=True):
    work.mkdir(parents=True, exist_ok=True)
    data, v3, v4 = find_data(inp, work), find_sub("v3", inp, work), find_sub("v4", inp, work)
    if verbose:
        n_img = len(list((data / "images").glob("*.png"))) if data else 0
        print("데이터:", data, f"| 이미지 {n_img}" if data else "| 없음 — 학습 데이터 데이터셋을 Add Input 하세요")
        for name, d in (("v3", v3), ("v4", v4)):
            print(f"{name}:", d, sorted(p.name for p in (d / "models").iterdir()) if d else "없음")
    return data, v3, v4
