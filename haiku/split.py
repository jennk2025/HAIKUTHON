"""검증 분할 생성. 데이터 점검 결과(data_audit.json)를 읽어 확정 중복 그룹을 같은 분할에 둔다.

실행 (저장소 최상위에서, haiku.audit 다음에):
    python -m haiku.split --data-dir haikuthon-train --audit reports/data_audit.json --out-dir splits

방법:
    1. 스타터 키트 train.py와 같은 무작위 분할(seed, 학습 80% / 검증 20%)에서 시작한다.
       확정 중복이 없으면 공식 분할과 똑같아서 공식 검증 점수(0.7839)와 바로 비교할 수 있다.
    2. 확정 중복 그룹(파일·픽셀·정규화 텍스트 완전 중복)이 학습·검증에 걸치면,
       그룹 전체를 구성원이 더 많은 쪽으로 옮긴다(같으면 학습 쪽).
    3. 근접 중복 후보·유사 텍스트 쌍이 학습·검증에 걸친 개수를 보고한다(옮기지 않음).

결과:
    splits/split_seed42.json   ID 목록 + 메타데이터 (실험에서 haiku.data.load_split으로 읽음)
    splits/train_ids.txt, splits/valid_ids.txt
    reports/split_report.md

규칙: 검증 ID는 학습과 증강 데이터 생성에 쓰지 않는다.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from haiku.data import load_rows


def official_train_ids(ids, seed, train_ratio_num=8, train_ratio_den=10):
    """스타터 키트 train.py load_and_split과 같은 계산 (ids는 숫자 오름차순)."""
    n_train = len(ids) * train_ratio_num // train_ratio_den
    return set(random.Random(seed).sample(ids, n_train))


def make_split(ids, groups, seed=42):
    ids = sorted(ids, key=int)
    official = official_train_ids(ids, seed)
    train = set(official)
    moved = []
    for g in groups:
        g = [i for i in g if i in set(ids)]
        in_t = [i for i in g if i in train]
        in_v = [i for i in g if i not in train]
        if in_t and in_v:
            to_train = len(in_t) >= len(in_v)
            for i in (in_v if to_train else in_t):
                (train.add if to_train else train.discard)(i)
                moved.append({"id": i, "to": "train" if to_train else "valid", "group": g})
    valid = [i for i in ids if i not in train]
    return sorted(train, key=int), valid, moved, train == official


def cross_split(pairs, train):
    cross = [p for p in pairs if (p["a"] in train) != (p["b"] in train)]
    return len(cross), cross


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("haikuthon-train"))
    ap.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("splits"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = load_rows(args.data_dir)
    ids = [r["id"] for r in rows]
    if len(ids) != audit["ids"]["n_unique"]:
        raise SystemExit("점검 결과와 데이터의 ID 수가 다릅니다. haiku.audit을 다시 실행하세요.")

    groups = audit["confirmed_duplicate_groups"]
    train_ids, valid_ids, moved, same_as_official = make_split(ids, groups, args.seed)
    train = set(train_ids)
    assert not train & set(valid_ids) and len(train) + len(valid_ids) == len(ids)

    n_img_cross, img_cross = cross_split(audit["near_image_pairs"], train)
    n_txt_cross, txt_cross = cross_split(audit["near_text_pairs"], train)
    groups_split = [g for g in groups if len({i in train for i in g}) > 1]

    meta = {
        "name": f"split_seed{args.seed}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "method": "스타터 키트 train.py 분할(random.Random(seed).sample, 80/20) + 확정 중복 그룹 같은 쪽 배치",
        "same_as_official_starterkit_split": same_as_official,
        "data_fingerprint": audit["data_fingerprint"],
        "n_train": len(train_ids), "n_valid": len(valid_ids),
        "confirmed_duplicate_groups": len(groups), "moved": moved,
        "cross_split": {
            "confirmed_groups_after_split": len(groups_split),
            "near_image_pairs": n_img_cross, "near_image_pairs_total": len(audit["near_image_pairs"]),
            "near_text_pairs": n_txt_cross, "near_text_pairs_total": len(audit["near_text_pairs"]),
        },
        "rules": ["검증 ID는 학습에 쓰지 않는다.", "검증 ID로 증강 데이터를 만들지 않는다.",
                  "검증 ID를 hard negative 채굴 대상으로 쓰지 않는다."],
        "train_ids": train_ids, "valid_ids": valid_ids,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"split_seed{args.seed}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.out_dir / "valid_ids.txt").write_text("\n".join(valid_ids) + "\n", encoding="utf-8")

    L = ["# 검증 분할 보고서\n",
         f"- 생성: {meta['created_at']} · seed {args.seed}",
         f"- 방법: {meta['method']}",
         f"- 데이터 지문: `{meta['data_fingerprint'][:16]}…` (data_audit.json과 같음)",
         f"- 결과: 학습 **{len(train_ids)}** / 검증 **{len(valid_ids)}**",
         f"- 스타터 키트 공식 분할과 동일: **{'예' if same_as_official else '아니오'}**",
         f"- 파일: `splits/split_seed{args.seed}.json`, `splits/train_ids.txt`, `splits/valid_ids.txt`\n",
         "## 중복·유사 데이터의 분할 상태\n",
         "| 항목 | 전체 | 학습·검증에 걸친 수 | 처리 |", "|---|---|---|---|",
         f"| 확정 중복 그룹 | {len(groups)} | {len(groups_split)} | 같은 쪽으로 옮김 ({len(moved)}개 ID 이동) |",
         f"| 이미지 근접 중복 후보 쌍 (pHash ≤ {audit['image_similarity']['threshold_phash']}) | {len(audit['near_image_pairs'])} | {n_img_cross} | 보고만 함 (확정 아님) |",
         f"| 텍스트 유사 쌍 (Jaccard ≥ {audit['text_similarity']['threshold']}) | {len(audit['near_text_pairs'])} | {n_txt_cross} | 보고만 함 (확정 아님) |",
         ""]
    if img_cross:
        L.append("학습·검증에 걸친 이미지 근접 중복 후보 (상위 10):\n")
        L += ["| A | B | pHash | dHash | A 분할 |", "|---|---|---|---|---|"]
        L += [f"| {p['a']} | {p['b']} | {p['phash_dist']} | {p['dhash_dist']} | {'학습' if p['a'] in train else '검증'} |"
              for p in img_cross[:10]]
        L.append("")
    L += ["## 확인 범위\n",
          "- 확정 중복: 파일 바이트, 디코딩 픽셀, 정규화 텍스트가 완전히 같은 경우",
          "- 유사 데이터: 지각 해시(pHash)와 단어 Jaccard로 찾을 수 있는 범위만 확인했다.",
          "- 임베딩 기반 의미 유사도(같은 장면을 다르게 그린 이미지 등)는 확인하지 않았다.\n",
          "## 사용 규칙\n"] + [f"- {r}" for r in meta["rules"]] + [
          "\n```python\nfrom haiku.data import load_rows, load_split\nrows = load_rows(DATA_DIR)\n"
          f"train_rows, valid_rows = load_split(REPO_DIR / 'splits/split_seed{args.seed}.json', rows)\n```"]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "split_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"분할 완료: 학습 {len(train_ids)} / 검증 {len(valid_ids)}, 공식 분할과 동일 {same_as_official}, "
          f"이동 {len(moved)}개, 걸친 근접 이미지 쌍 {n_img_cross}, 걸친 유사 텍스트 쌍 {n_txt_cross}")


if __name__ == "__main__":
    main()
