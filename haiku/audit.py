"""HAIKUTHON Train 데이터 점검. 원본 파일은 읽기만 하고 삭제·수정하지 않는다.

실행 (저장소 최상위에서):
    python -m haiku.audit --data-dir haikuthon-train --out-dir reports

결과:
    reports/data_audit.md     사람이 읽는 보고서
    reports/data_audit.json   요약, 문제 목록, 중복 그룹, 유사 쌍 (split 단계가 읽음)
    reports/image_index.csv   이미지별 형식·크기·해시
    reports/near_dup_image_pairs.png  이미지 근접 중복 후보 상위 쌍 (육안 확인용)

검사 범위:
    - 완전 중복: 파일 바이트(SHA-256), 디코딩한 픽셀(MD5), 텍스트(원문 / 정규화)
    - 이미지 유사: 지각 해시(pHash, dHash) 해밍 거리. 의미 수준(임베딩) 유사도는 검사하지 않는다.
    - 텍스트 유사: 단어 집합 Jaccard, 단어 순서만 다른 문장
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ID_RE = re.compile(r"^\d{4}$")
EXPECTED_KEYS = {"id", "prompts"}
PHASH_NEAR = 8            # 이미지 근접 중복 후보: pHash 해밍 거리 ≤ 8 (64비트 중)
TEXT_JACCARD_NEAR = 0.9   # 텍스트 유사 쌍: 단어 집합 Jaccard ≥ 0.9
NEAR_UNIFORM_STD = 3.0    # 픽셀 표준편차가 이보다 작으면 거의 단색 이미지로 표시
MAX_PAIRS_SAVED = 20000


# ----------------------------------------------------------------------------- 해시
def _dct_matrix(n=32):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_DCT32 = _dct_matrix(32)


def _bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def dhash(gray):
    a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return _bits_to_int((a[:, 1:] > a[:, :-1]).flatten())


def phash(gray):
    a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT32 @ a @ _DCT32.T)[:8, :8].flatten()
    return _bits_to_int(low > np.median(low[1:]))


def _image_worker(path):
    path = Path(path)
    rec = {"id": path.stem, "file": path.name}
    try:
        data = path.read_bytes()
        rec["bytes"] = len(data)
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()                      # 구조 검사
        with Image.open(io.BytesIO(data)) as im:
            rec["format"], rec["mode"] = im.format, im.mode
            rec["width"], rec["height"] = im.size
            im.load()                        # 전체 디코딩
            rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        rec["pixel_md5"] = hashlib.md5(arr.tobytes()).hexdigest()
        rec["pixel_mean"] = round(float(arr.mean()), 2)
        rec["pixel_std"] = round(float(arr.std()), 2)
        gray = rgb.convert("L")
        rec["phash"] = phash(gray)
        rec["dhash"] = dhash(gray)
        rec["ok"] = True
    except Exception as e:  # 손상 이미지는 기록만 한다
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def hamming_neighbors(hashes, max_dist, block=256):
    """64비트 해시 목록에서 거리 ≤ max_dist인 쌍 (i<j, dist)과 각 원소의 최근접 거리."""
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    nearest = np.full(n, 64, dtype=np.int64)
    pairs = []
    for s in range(0, n, block):
        x = np.ascontiguousarray(h[s:s + block, None] ^ h[None, :])
        d = lut[x.view(np.uint8)].reshape(x.shape[0], n, 8).sum(-1).astype(np.int64)
        rows = np.arange(x.shape[0])
        d[rows, rows + s] = 999                       # 자기 자신 제외
        nearest[s:s + x.shape[0]] = d.min(1)
        ii, jj = np.nonzero(d <= max_dist)
        for a, b in zip(ii + s, jj):
            if a < b:
                pairs.append((int(a), int(b), int(d[a - s, b])))
    return pairs, nearest


def popcount64(a, b):
    return bin(int(a) ^ int(b)).count("1")


# ----------------------------------------------------------------------------- 텍스트
def normalize_text(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def words(t):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower())


def text_jaccard_neighbors(texts, threshold):
    vocab = {}
    sets = [set(words(t)) for t in texts]
    for s in sets:
        for w in s:
            vocab.setdefault(w, len(vocab))
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, s in enumerate(sets):
        x[i, [vocab[w] for w in s]] = 1.0
    sizes = x.sum(1)
    inter = x @ x.T
    jac = inter / (sizes[:, None] + sizes[None, :] - inter)
    np.fill_diagonal(jac, -1.0)
    nearest = jac.max(1)
    ii, jj = np.nonzero(np.triu(jac >= threshold, k=1))
    pairs = [(int(a), int(b), round(float(jac[a, b]), 4)) for a, b in zip(ii, jj)]
    return pairs, nearest, len(vocab)


def groups_by(keys_by_id):
    g = defaultdict(list)
    for i, k in keys_by_id.items():
        g[k].append(i)
    return sorted((sorted(v) for v in g.values() if len(v) > 1), key=lambda v: v[0])


def union_groups(*group_lists):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for groups in group_lists:
        for g in groups:
            for other in g[1:]:
                parent[find(other)] = find(g[0])
    merged = defaultdict(list)
    for a in list(parent):
        merged[find(a)].append(a)
    return sorted((sorted(v) for v in merged.values() if len(v) > 1), key=lambda v: v[0])


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "mean": round(float(v.mean()), 2), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def bucket_counts(values, edges):
    """edges=[0,4,8] → {"0": n, "1-4": n, "5-8": n, ">8": n} (정수 거리용)"""
    out, lo = {}, None
    v = np.asarray(values)
    for e in edges:
        label = f"{e}" if lo is None else (f"{lo + 1}-{e}" if lo + 1 != e else f"{e}")
        mask = (v <= e) if lo is None else ((v > lo) & (v <= e))
        out[label] = int(mask.sum())
        lo = e
    out[f">{edges[-1]}"] = int((v > edges[-1]).sum())
    return out


# ----------------------------------------------------------------------------- 점검
def audit(data_dir, out_dir, workers=None, phash_near=PHASH_NEAR, text_near=TEXT_JACCARD_NEAR):
    t_start = time.time()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    issues = []

    def issue(severity, check, detail, ids=()):
        ids = sorted(ids)
        issues.append({"severity": severity, "check": check, "detail": detail, "count": len(ids), "ids": ids[:50]})

    # 1. 구조 --------------------------------------------------------------
    images_dir = data_dir / "images"
    prompts_path = data_dir / "prompts.jsonl"
    if not prompts_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f"{data_dir}에 prompts.jsonl 또는 images/가 없습니다.")
    files = [p for p in images_dir.iterdir() if p.is_file()]
    structure = {
        "data_dir": str(data_dir.resolve()),
        "top_level": sorted(p.name + ("/" if p.is_dir() else "") for p in data_dir.iterdir()),
        "images_dir_extensions": dict(Counter(p.suffix.lower() for p in files)),
        "images_dir_non_png": sorted(p.name for p in files if p.suffix.lower() != ".png"),
    }

    # 2. 텍스트 ------------------------------------------------------------
    raw_bytes = prompts_path.read_bytes()
    rows, blank_lines = [], []
    for ln, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            blank_lines.append(ln)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            issue("error", "JSONL 파싱", f"{ln}행: {e}")
            continue
        if set(obj) != EXPECTED_KEYS:
            issue("warn", "필드 구성", f"{ln}행 키 {sorted(obj)} (예상 {sorted(EXPECTED_KEYS)})")
        rid, prompt = obj.get("id"), obj.get("prompts")
        if not isinstance(rid, str):
            issue("error", "ID 형식", f"{ln}행 id가 문자열이 아님: {rid!r} (앞자리 0 손실 위험)")
            rid = str(rid)
        if not isinstance(prompt, str):
            issue("error", "설명 형식", f"{ln}행 prompts가 문자열이 아님: {type(prompt).__name__}", [rid])
            prompt = "" if prompt is None else str(prompt)
        rows.append({"line": ln, "id": rid, "prompt": prompt})
    if blank_lines:
        issue("info", "빈 줄", f"prompts.jsonl 빈 줄 {len(blank_lines)}개 (행 {blank_lines[:10]})")

    ids = [r["id"] for r in rows]
    id_counts = Counter(ids)
    bad_ids = [i for i in id_counts if not ID_RE.match(i)]
    dup_ids = [i for i, c in id_counts.items() if c > 1]
    if bad_ids:
        issue("error", "ID 형식", "네 자리 숫자 문자열이 아닌 ID", bad_ids)
    if dup_ids:
        issue("error", "중복 ID", "prompts.jsonl에 같은 ID가 여러 번 나옴", dup_ids)
    numeric = sorted(int(i) for i in id_counts if ID_RE.match(i))
    consecutive = bool(numeric) and numeric == list(range(numeric[0], numeric[0] + len(numeric)))

    prompts = {r["id"]: r["prompt"] for r in rows}
    empty = [i for i, p in prompts.items() if not p.strip()]
    edge_ws = [i for i, p in prompts.items() if p != p.strip()]
    multi_space = [i for i, p in prompts.items() if "  " in p]
    non_ascii = [i for i, p in prompts.items() if any(ord(c) > 127 for c in p)]
    newline = [i for i, p in prompts.items() if "\n" in p]
    control = [i for i, p in prompts.items() if any(ord(c) < 32 and c != "\n" for c in p)]
    if empty:
        issue("error", "빈 설명", "비어 있거나 공백뿐인 설명", empty)
    if edge_ws:
        issue("info", "앞뒤 공백", "설명 앞뒤에 공백이 있음 (원본 유지)", edge_ws)
    if multi_space:
        issue("info", "연속 공백", "설명 안에 공백이 연속으로 있음 (원본 유지)", multi_space)
    if non_ascii:
        issue("info", "비ASCII 문자", "영문 외 문자가 들어 있음", non_ascii)
    if newline:
        issue("info", "문단 줄바꿈", "설명 안에 줄바꿈(\\n\\n)이 있음. 문단 구분으로 보이며 원본 유지", newline)
    if control:
        issue("warn", "제어 문자", "줄바꿈 외 제어 문자(탭 등)가 들어 있음", control)

    # metadata.jsonl (있을 때만)
    meta_path = images_dir / "metadata.jsonl"
    metadata = {"present": meta_path.is_file()}
    if meta_path.is_file():
        meta = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        mismatch = [m.get("id") for m in meta
                    if prompts.get(m.get("id")) != m.get("prompts") or m.get("file_name") != f"{m.get('id')}.png"]
        metadata.update(rows=len(meta), matches_prompts_jsonl=not mismatch and len(meta) == len(prompts))
        if mismatch or len(meta) != len(prompts):
            issue("warn", "metadata.jsonl 불일치", f"prompts.jsonl과 다른 행 {len(mismatch)}개, 행 수 {len(meta)}",
                  [m for m in mismatch if m])

    # 3. 이미지 ↔ ID 대응 --------------------------------------------------
    pngs = {p.stem: p for p in files if p.suffix.lower() == ".png"}
    missing = [i for i in id_counts if i not in pngs]
    extra = [s for s in pngs if s not in id_counts]
    if missing:
        issue("error", "누락 이미지", "설명은 있는데 이미지 파일이 없음", missing)
    if extra:
        issue("warn", "짝 없는 이미지", "이미지 파일은 있는데 설명이 없음", extra)
    upper_ext = [p.name for p in files if p.suffix == ".PNG"]
    if upper_ext:
        issue("warn", "확장자 대소문자", "확장자가 .PNG (경로 처리 주의)", [Path(n).stem for n in upper_ext])

    # 4. 이미지 디코딩·해시 (병렬) ----------------------------------------
    order = sorted(pngs, key=lambda s: (not ID_RE.match(s), int(s) if ID_RE.match(s) else 0, s))
    workers = workers or os.cpu_count() or 1
    t_img = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            recs = list(ex.map(_image_worker, [str(pngs[s]) for s in order], chunksize=16))
    else:
        recs = [_image_worker(str(pngs[s])) for s in order]
    image_sec = round(time.time() - t_img, 1)
    ok = [r for r in recs if r["ok"]]
    broken = [r for r in recs if not r["ok"]]
    if broken:
        issue("error", "손상 이미지", "열거나 끝까지 디코딩할 수 없음: " + "; ".join(f"{r['id']}: {r['error']}" for r in broken[:5]),
              [r["id"] for r in broken])
    wrong_format = [r["id"] for r in ok if r["format"] != "PNG"]
    if wrong_format:
        issue("warn", "형식 불일치", "확장자는 .png인데 실제 형식이 PNG가 아님", wrong_format)
    near_uniform = [r["id"] for r in ok if r["pixel_std"] < NEAR_UNIFORM_STD]
    if near_uniform:
        issue("warn", "거의 단색 이미지", f"픽셀 표준편차 < {NEAR_UNIFORM_STD}", near_uniform)
    images = {
        "n_files": len(recs), "n_decoded_ok": len(ok), "n_broken": len(broken), "decode_seconds": image_sec, "workers": workers,
        "formats": dict(Counter(r["format"] for r in ok)),
        "modes": dict(Counter(r["mode"] for r in ok)),
        "sizes": {f"{w}x{h}": c for (w, h), c in Counter((r["width"], r["height"]) for r in ok).items()},
        "file_bytes": describe([r["bytes"] for r in ok]) if ok else None,
        "pixel_std": describe([r["pixel_std"] for r in ok]) if ok else None,
    }
    if len(images["sizes"]) > 1:
        issue("info", "이미지 크기 여러 종류", f"크기 분포 {images['sizes']}")

    # 5. 완전 중복 --------------------------------------------------------
    rec_by_id = {r["id"]: r for r in ok}
    text_ids = [i for i in prompts if i in rec_by_id]  # 이미지와 짝이 맞는 것만 유사도 계산
    exact = {
        "image_file_sha256": groups_by({i: r["sha256"] for i, r in rec_by_id.items()}),
        "image_pixels_md5": groups_by({i: r["pixel_md5"] for i, r in rec_by_id.items()}),
        "text_raw": groups_by({i: prompts[i] for i in prompts}),
        "text_normalized": groups_by({i: normalize_text(prompts[i]) for i in prompts}),
    }
    pair_dup = groups_by({i: (rec_by_id[i]["pixel_md5"], normalize_text(prompts[i])) for i in text_ids})
    exact["image_and_text_pair"] = pair_dup
    labels = {"image_file_sha256": "이미지 파일 완전 중복", "image_pixels_md5": "이미지 픽셀 완전 중복",
              "text_raw": "텍스트 완전 중복", "text_normalized": "텍스트 정규화 후 중복(대소문자·문장부호·공백 무시)",
              "image_and_text_pair": "이미지·텍스트 쌍 전체 중복"}
    for k, groups in exact.items():
        if groups:
            issue("warn", labels[k], f"{len(groups)}개 그룹", [i for g in groups for i in g])
    confirmed_groups = union_groups(exact["image_file_sha256"], exact["image_pixels_md5"], exact["text_normalized"])

    # 6. 이미지 유사 (지각 해시) ------------------------------------------
    img_ids = [r["id"] for r in ok]
    ph_pairs, ph_nearest = hamming_neighbors([rec_by_id[i]["phash"] for i in img_ids], phash_near)
    near_img_pairs = sorted(
        ({"a": img_ids[a], "b": img_ids[b], "phash_dist": d,
          "dhash_dist": popcount64(rec_by_id[img_ids[a]]["dhash"], rec_by_id[img_ids[b]]["dhash"])}
         for a, b, d in ph_pairs),
        key=lambda p: (p["phash_dist"], p["dhash_dist"], p["a"], p["b"]))
    image_similarity = {
        "method": "pHash(32x32 DCT 저주파 8x8, 64비트) 해밍 거리, 후보 쌍에는 dHash 거리도 함께 기록",
        "threshold_phash": phash_near,
        "n_candidate_pairs": len(near_img_pairs),
        "n_images_in_pairs": len({p["a"] for p in near_img_pairs} | {p["b"] for p in near_img_pairs}),
        "nearest_phash_distance_buckets": bucket_counts(ph_nearest, [0, 4, 8, 12, 16, 20]),
        "not_checked": "의미 수준(임베딩) 유사도는 검사하지 않음. 같은 장면을 다른 구도로 그린 이미지는 해시로 잡히지 않는다.",
    }

    # 7. 텍스트 유사 ------------------------------------------------------
    tx_ids = list(prompts)
    tx_pairs, tx_nearest, vocab_size = text_jaccard_neighbors([prompts[i] for i in tx_ids], text_near)
    near_txt_pairs = sorted(({"a": tx_ids[a], "b": tx_ids[b], "jaccard": j} for a, b, j in tx_pairs),
                            key=lambda p: (-p["jaccard"], p["a"], p["b"]))
    bag = groups_by({i: " ".join(sorted(words(prompts[i]))) for i in prompts})
    norm_groups = {tuple(g) for g in exact["text_normalized"]}
    reorder_only = [g for g in bag if tuple(g) not in norm_groups]
    if reorder_only:
        issue("info", "같은 단어 집합 텍스트", f"{len(reorder_only)}개 그룹. 속성·관계 결합이 달라 중복으로 확정하지 않음",
              [i for g in reorder_only for i in g])
    nj = np.asarray(tx_nearest)
    text_similarity = {
        "method": "단어 집합 Jaccard (소문자, 영숫자 단어)",
        "vocab_size": vocab_size,
        "threshold": text_near,
        "n_pairs_at_or_above_threshold": len(near_txt_pairs),
        "nearest_jaccard_buckets": {
            "1.0": int((nj >= 1.0).sum()), "0.9-1.0": int(((nj >= 0.9) & (nj < 1.0)).sum()),
            "0.8-0.9": int(((nj >= 0.8) & (nj < 0.9)).sum()), "0.7-0.8": int(((nj >= 0.7) & (nj < 0.8)).sum()),
            "<0.7": int((nj < 0.7).sum())},
        "same_words_different_order_groups": reorder_only,
    }

    # 8. 텍스트 통계 ------------------------------------------------------
    text_stats = {
        "n": len(prompts), "chars": describe([len(p) for p in prompts.values()]),
        "words": describe([len(p.split()) for p in prompts.values()]),
        "sentences": describe([len(re.findall(r"[.!?](?:\s|$)", p)) for p in prompts.values()]),
    }

    fingerprint = hashlib.sha256(raw_bytes + "".join(rec_by_id[i]["sha256"] for i in sorted(rec_by_id)).encode()).hexdigest()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "seconds": None},
        "data_fingerprint": fingerprint,
        "structure": structure,
        "ids": {"n_rows": len(rows), "n_unique": len(id_counts), "all_4digit_str": not bad_ids,
                "consecutive": consecutive, "range": [f"{numeric[0]:04d}", f"{numeric[-1]:04d}"] if numeric else None,
                "n_duplicate_ids": len(dup_ids), "n_missing_images": len(missing), "n_images_without_text": len(extra)},
        "metadata_jsonl": metadata,
        "text": text_stats,
        "images": images,
        "exact_duplicates": {k: {"n_groups": len(v), "groups": v} for k, v in exact.items()},
        "confirmed_duplicate_groups": confirmed_groups,
        "image_similarity": image_similarity,
        "text_similarity": text_similarity,
        "near_image_pairs": near_img_pairs[:MAX_PAIRS_SAVED],
        "near_text_pairs": near_txt_pairs[:MAX_PAIRS_SAVED],
        "pairs_truncated": len(near_img_pairs) > MAX_PAIRS_SAVED or len(near_txt_pairs) > MAX_PAIRS_SAVED,
        "issues": issues,
    }
    summary["runtime"]["seconds"] = round(time.time() - t_start, 1)

    (out_dir / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / "image_index.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["id", "file", "ok", "format", "mode", "width", "height", "bytes", "sha256", "pixel_md5",
                "pixel_mean", "pixel_std", "phash", "dhash", "error"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            r = dict(r)
            for k in ("phash", "dhash"):
                if k in r:
                    r[k] = f"{r[k]:016x}"
            w.writerow(r)
    _save_pair_montage(near_img_pairs[:12], pngs, prompts, out_dir / "near_dup_image_pairs.png")
    (out_dir / "data_audit.md").write_text(render_markdown(summary, prompts), encoding="utf-8")
    return summary


def _save_pair_montage(pairs, pngs, prompts, path, thumb=224):
    if not pairs:
        if path.exists():
            path.unlink()
        return
    from PIL import ImageDraw
    pad, label_h = 8, 22
    w = 2 * thumb + 3 * pad
    h = len(pairs) * (thumb + label_h + pad) + pad
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for p in pairs:
        for k, key in enumerate(("a", "b")):
            with Image.open(pngs[p[key]]) as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (pad + k * (thumb + pad), y + label_h))
        draw.text((pad, y + 4), f"{p['a']} vs {p['b']}  pHash {p['phash_dist']}  dHash {p['dhash_dist']}", fill="black")
        y += thumb + label_h + pad
    sheet.save(path)


# ----------------------------------------------------------------------------- 보고서
def render_markdown(s, prompts):
    sev_order = {"error": 0, "warn": 1, "info": 2}
    sev_ko = {"error": "오류", "warn": "주의", "info": "참고"}
    ids, im, tx = s["ids"], s["images"], s["text"]
    L = []
    a = L.append
    a("# Train 데이터 점검 보고서\n")
    a(f"- 생성: {s['generated_at']} · 실행 시간 {s['runtime']['seconds']}초 (이미지 디코딩 {im['decode_seconds']}초, worker {im['workers']})")
    a(f"- 데이터: `{s['structure']['data_dir']}`")
    a(f"- 데이터 지문(SHA-256): `{s['data_fingerprint'][:16]}…` — 분할 파일과 같은 데이터인지 확인할 때 쓴다")
    a("- 원본 파일은 읽기만 했다. 삭제·수정한 파일은 없다.\n")

    a("## 1. 요약\n")
    a("| 점검 | 결과 |")
    a("|---|---|")
    a(f"| 텍스트 행 / 고유 ID | {ids['n_rows']} / {ids['n_unique']} |")
    a(f"| ID 형식 | 네 자리 문자열 {'예' if ids['all_4digit_str'] else '아니오'}, 연속 {'예' if ids['consecutive'] else '아니오'}, 범위 {ids['range']} |")
    a(f"| 중복 ID | {ids['n_duplicate_ids']} |")
    a(f"| 누락 이미지 / 설명 없는 이미지 | {ids['n_missing_images']} / {ids['n_images_without_text']} |")
    empty = next((i['count'] for i in s['issues'] if i['check'] == '빈 설명'), 0)
    a(f"| 빈 설명 | {empty} |")
    a(f"| 이미지 디코딩 성공 / 손상 | {im['n_decoded_ok']} / {im['n_broken']} |")
    a(f"| 이미지 형식 · 모드 · 크기 | {im['formats']} · {im['modes']} · {im['sizes']} |")
    for k, v in s["exact_duplicates"].items():
        a(f"| 완전 중복 `{k}` | {v['n_groups']}개 그룹 |")
    a(f"| 확정 중복 그룹 (분할 때 같은 쪽에 묶음) | {len(s['confirmed_duplicate_groups'])}개 |")
    isim, tsim = s["image_similarity"], s["text_similarity"]
    a(f"| 이미지 근접 중복 후보 (pHash ≤ {isim['threshold_phash']}) | {isim['n_candidate_pairs']}쌍 |")
    a(f"| 텍스트 유사 쌍 (Jaccard ≥ {tsim['threshold']}) | {tsim['n_pairs_at_or_above_threshold']}쌍 |")
    a("")

    a("## 2. 실제 디렉터리 구조\n")
    a(f"- 최상위: {', '.join(f'`{x}`' for x in s['structure']['top_level'])}")
    a(f"- `images/` 확장자: {s['structure']['images_dir_extensions']}")
    a(f"- `images/`의 PNG 외 파일: {s['structure']['images_dir_non_png'] or '없음'}")
    m = s["metadata_jsonl"]
    a(f"- `images/metadata.jsonl`: " + ("없음 (HF에서 `images/*.png`, `prompts.jsonl`만 받은 경우)" if not m["present"]
                                       else f"{m['rows']}행, prompts.jsonl과 일치 {'예' if m['matches_prompts_jsonl'] else '아니오'}"))
    a("- `prompts.jsonl` 형식: 한 줄에 `{\"id\": \"0000\", \"prompts\": \"...\"}`. ID는 문자열로 읽어서 앞자리 0을 유지한다.\n")

    a("## 3. 텍스트 통계\n")
    a("| 항목 | 최소 | 평균 | 중앙값 | 95% | 최대 |")
    a("|---|---|---|---|---|---|")
    for k, name in (("chars", "글자 수"), ("words", "단어 수"), ("sentences", "문장 수")):
        d = tx[k]
        a(f"| {name} | {d['min']:.0f} | {d['mean']} | {d['p50']:.0f} | {d['p95']:.0f} | {d['max']:.0f} |")
    a("")

    a("## 4. 이미지 통계\n")
    fb, ps = im["file_bytes"], im["pixel_std"]
    a(f"- 파일 크기(바이트): 최소 {fb['min']:.0f}, 중앙값 {fb['p50']:.0f}, 최대 {fb['max']:.0f}")
    a(f"- 픽셀 표준편차: 최소 {ps['min']}, 중앙값 {ps['p50']}, 최대 {ps['max']} (거의 단색 기준 < {NEAR_UNIFORM_STD})\n")

    a("## 5. 중복과 유사 데이터\n")
    a("### 완전 중복 (확정)\n")
    a("| 종류 | 그룹 수 | 예시 |")
    a("|---|---|---|")
    for k, v in s["exact_duplicates"].items():
        a(f"| `{k}` | {v['n_groups']} | {', '.join('/'.join(g) for g in v['groups'][:3]) or '—'} |")
    a("")
    a("### 이미지 근접 중복 후보 (확정 아님)\n")
    a("- 지각 해시는 구도·배치가 비슷하면 가깝게 나온다. 색·재질처럼 속성 하나만 다른 이미지도 후보에 들어오므로 "
      "후보는 반드시 `near_dup_image_pairs.png`로 육안 확인한다.")
    a(f"- 방법: {isim['method']}")
    a(f"- 각 이미지의 가장 가까운 다른 이미지까지 pHash 거리 분포: {isim['nearest_phash_distance_buckets']}")
    a(f"- 후보 {isim['n_candidate_pairs']}쌍, 관련 이미지 {isim['n_images_in_pairs']}장")
    if s["near_image_pairs"]:
        a("- 상위 쌍 (육안 확인용 그림: `near_dup_image_pairs.png`):\n")
        a("| A | B | pHash | dHash |")
        a("|---|---|---|---|")
        for p in s["near_image_pairs"][:10]:
            a(f"| {p['a']} | {p['b']} | {p['phash_dist']} | {p['dhash_dist']} |")
    a(f"- **검사하지 않은 것**: {isim['not_checked']}\n")
    a("### 텍스트 유사\n")
    a(f"- 방법: {tsim['method']}, 어휘 {tsim['vocab_size']}개")
    a(f"- 각 문장의 가장 비슷한 다른 문장까지 Jaccard 분포: {tsim['nearest_jaccard_buckets']}")
    a(f"- Jaccard ≥ {tsim['threshold']} 쌍: {tsim['n_pairs_at_or_above_threshold']}개")
    a(f"- 같은 단어 집합(순서·결합만 다름) 그룹: {len(tsim['same_words_different_order_groups'])}개")
    a("- 주의: 단어 집합이 같아도 속성이 다른 객체에 붙거나 관계 방향이 바뀌면 다른 장면이다. 이 지표는 중복 판정에 쓰지 않는다.")
    for p in s["near_text_pairs"][:3]:
        a(f"\n  - `{p['a']}` vs `{p['b']}` (Jaccard {p['jaccard']})")
        a(f"    - {p['a']}: {prompts[p['a']]}")
        a(f"    - {p['b']}: {prompts[p['b']]}")
    a("")

    a("## 6. 발견한 문제 목록\n")
    if not s["issues"]:
        a("발견한 문제 없음.\n")
    else:
        a("| 수준 | 점검 | 내용 | 개수 | ID 예시 |")
        a("|---|---|---|---|---|")
        for i in sorted(s["issues"], key=lambda x: sev_order[x["severity"]]):
            a(f"| {sev_ko[i['severity']]} | {i['check']} | {i['detail']} | {i['count']} | {', '.join(i['ids'][:8])} |")
        a("")
    a("## 7. 이번 점검의 범위\n")
    a("- 수행: ID 대응, 앞자리 0 유지, 중복 ID, 빈 설명, 누락·짝 없는 이미지, 전체 디코딩(손상), 형식·모드·크기 분포, "
      "파일·픽셀·텍스트 완전 중복, pHash/dHash 이미지 근접 중복 후보, 단어 Jaccard 텍스트 유사도")
    a("- 수행하지 않음: 임베딩(CLIP/SigLIP) 기반 의미 유사 이미지 탐지, 이미지와 설명의 내용 일치 여부 검사")
    a("- 원본 텍스트·이미지는 수정하지 않았다.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("haikuthon-train"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--workers", type=int, default=None, help="이미지 디코딩 프로세스 수 (기본: CPU 수)")
    ap.add_argument("--phash-near", type=int, default=PHASH_NEAR)
    ap.add_argument("--text-near", type=float, default=TEXT_JACCARD_NEAR)
    args = ap.parse_args()
    s = audit(args.data_dir, args.out_dir, args.workers, args.phash_near, args.text_near)
    n_err = sum(i["severity"] == "error" for i in s["issues"])
    print(f"점검 완료: 문제 {len(s['issues'])}건 (오류 {n_err}건), {s['runtime']['seconds']}초 → {args.out_dir}/data_audit.md")


if __name__ == "__main__":
    main()
