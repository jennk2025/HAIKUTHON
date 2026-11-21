"""v4가 끝까지 틀리는 문장은 어떤 구조인가: 검증 질의별 순위에 문장 구조 특징을 붙여 범주별 1위 정답률을 본다 (GPU 불필요).

- 특징은 질의 문장 하나에서 규칙 파서로만 계산한다 (분석용, 순위 계산에는 쓰지 않음).
- 범주마다 질의 수를 함께 적어 드문 표현에 끌려가지 않게 하고, 문장 길이 구간별로도 나눠 본다.
"""

import re

from haiku.siblings import parse_prompt

REL_KO = {"left_of": "좌우", "above": "상하", "next_to": "옆(next to)", "on": "위에 놓임(on)", "inside": "안(inside)",
          "holds": "들고 있음", "wears": "착용", "rides": "타고 있음"}


def features(prompt):
    p = parse_prompt(prompt)
    multi = {f.split(":", 1)[1] for f in p["flags"] if f.startswith("multi_instance:")}
    per_obj = [sum(len(v) for v in o["attrs"].values()) for o in p["objects"].values()]
    rel_kinds = {r for r, _, _ in p["relations"]}
    return {"words": len(prompt.split()), "n_obj": len(p["objects"]), "multi": bool(multi),
            "multi_in_rel": any(x in multi or y in multi for _, x, y in p["relations"]),
            "max_attrs_obj": max(per_obj) if per_obj else 0, "rel_kinds": rel_kinds, "n_rel": len(p["relations"]),
            "both": bool(re.search(r"\bboth\b", prompt, re.I)), "one_of": bool(re.search(r"\bone of\b", prompt, re.I))}


def _line(name, idx, ranks, base=None):
    n = len(idx)
    if not n:
        return f"  {name:28s} 질의    0"
    r1 = sum(ranks[i] == 1 for i in idx) / n
    s = f"  {name:28s} 질의 {n:4d}  1위 정답률 {r1:.3f}"
    if base is not None:
        fx = sum(base[i] > 1 and ranks[i] == 1 for i in idx)
        br = sum(base[i] == 1 and ranks[i] > 1 for i in idx)
        s += f"  | v3 대비 고침 {fx:3d} 망침 {br:3d}"
    return s


def report(rows, per_query, variant="v3+OWL(v4값 0.1/0.06)", base_variant="v3"):
    """per_query: run_vs_v3 결과의 'per_query' ({'q': [질의 ID], 'ranks': {변형 이름: [순위]}})."""
    by_id = {r["id"]: r for r in rows}
    q = per_query["q"]
    ranks, base = per_query["ranks"][variant], per_query["ranks"].get(base_variant)
    F = [features(by_id[i]["prompt"]) for i in q]
    allidx = list(range(len(q)))
    L = [f"===== 문장 구조별 1위 정답률 ({variant}, 검증 {len(q)}질의) =====", _line("전체", allidx, ranks, base)]
    w = sorted(f["words"] for f in F)
    cut = [w[len(w) // 3], w[2 * len(w) // 3]]
    band = lambda f: 0 if f["words"] < cut[0] else (1 if f["words"] < cut[1] else 2)
    names = [f"짧음(<{cut[0]}단어)", f"중간({cut[0]}~{cut[1] - 1})", f"긺(≥{cut[1]})"]
    L.append("[문장 길이]")
    L += [_line(names[b], [i for i in allidx if band(F[i]) == b], ranks, base) for b in range(3)]
    L.append("[물체 종류 수]")
    for lab, cond in (("2개", lambda f: f["n_obj"] <= 2), ("3개", lambda f: f["n_obj"] == 3), ("4개 이상", lambda f: f["n_obj"] >= 4)):
        L.append(_line(lab, [i for i in allidx if cond(F[i])], ranks, base))
    L.append("[한 물체에 붙은 속성 수(최대)]")
    for lab, cond in (("1개 이하", lambda f: f["max_attrs_obj"] <= 1), ("2개", lambda f: f["max_attrs_obj"] == 2),
                      ("3개 이상", lambda f: f["max_attrs_obj"] >= 3)):
        L.append(_line(lab, [i for i in allidx if cond(F[i])], ranks, base))
    L.append("[같은 종류 물체가 여러 개 → 대상을 먼저 특정해야 함]")
    L.append(_line("있음", [i for i in allidx if F[i]["multi"]], ranks, base))
    L.append(_line("없음", [i for i in allidx if not F[i]["multi"]], ranks, base))
    L.append("  (길이 구간별 있음 / 없음 1위 정답률)  " + " | ".join(
        f"{names[b]}: " + " / ".join(
            (lambda idx: f"{sum(ranks[i] == 1 for i in idx) / len(idx):.2f}(n={len(idx)})" if idx else "-")(
                [i for i in allidx if band(F[i]) == b and F[i]["multi"] == m]) for m in (True, False))
        for b in range(3)))
    L.append("[관계 종류 (한 질의가 여러 범주에 들 수 있음)]")
    for k, ko in REL_KO.items():
        L.append(_line(ko, [i for i in allidx if k in F[i]["rel_kinds"]], ranks, base))
    L.append("[짧은 한정 표현]")
    L.append(_line("both", [i for i in allidx if F[i]["both"]], ranks, base))
    L.append(_line("one of", [i for i in allidx if F[i]["one_of"]], ranks, base))
    return "\n".join(L)


def by_relation(rows, per_query, variants=("v3", "v3+OWL(v4값 0.1/0.06)", "v3+OWL확장")):
    """관계 종류별로 변형들의 1위 정답률을 나란히 본다 (질의 수 포함)."""
    by_id = {r["id"]: r for r in rows}
    q = per_query["q"]
    F = [features(by_id[i]["prompt"]) for i in q]
    variants = [v for v in variants if v in per_query["ranks"]]
    L = ["===== 관계 종류별 1위 정답률: " + " → ".join(variants) + " ====="]
    for k, ko in REL_KO.items():
        idx = [i for i in range(len(q)) if k in F[i]["rel_kinds"]]
        if not idx:
            continue
        accs = [sum(per_query["ranks"][v][i] == 1 for i in idx) / len(idx) for v in variants]
        L.append(f"  {ko:16s} 질의 {len(idx):4d}  " + " → ".join(f"{a:.3f}" for a in accs))
    return "\n".join(L)


def diagnose(rows, per_query, variant="v3+OWL확장", k_blip=5, n_cases=30, seed=0):
    """정답을 어느 단계에서 잃는가 + 최종 2위로 남은 사례의 원인 분류 (검증 데이터 분석용, 순위 계산에는 쓰지 않음).
    원인은 정답 문장과 1위 오답 이미지의 원래 문장을 규칙 파서로 비교해 정한다."""
    import random
    from collections import Counter

    from haiku.siblings import CAT_KO, compare, fmt_diffs, parse_prompt
    by_id = {r["id"]: r for r in rows}
    q, R = per_query["q"], per_query["ranks"]
    fin, s, c = R[variant], R["search"], R.get("cond")
    n = len(q)
    L = [f"===== 정답을 어디서 잃는가 ({variant}, 검증 {n}질의) ====="]
    L.append(f"  검색 상위 10장 밖 (첫 검색에서 놓침)               {sum(r > 10 for r in s):4d}")
    if c:
        L.append(f"  상위 10장 안이었지만 조건 재정렬 후 BLIP이 보는 {k_blip}장 밖 {sum(a <= 10 and b > k_blip for a, b in zip(s, c)):4d}")
    L.append(f"  최종 1위 {sum(r == 1 for r in fin)} | 2위 {sum(r == 2 for r in fin)} | 3~5위 {sum(2 < r <= 5 for r in fin)} | 5위 밖 {sum(r > 5 for r in fin)}")
    if "owl_ext_applicable" in per_query:
        app = per_query["owl_ext_applicable"]
        L.append(f"  OWL 적용 가능 질의 {sum(app)} | 그중 순서를 바꾼 질의 {sum(per_query['owl_ext_swapped'])}")
    if "top2_v5" not in per_query:
        return "\n".join(L)
    labels, details = Counter(), []
    for i in range(n):
        if fin[i] != 2:
            continue
        gold, wrong = q[i], per_query["top2_v5"][i][0]
        pa, pb = parse_prompt(by_id[gold]["prompt"]), parse_prompt(by_id[wrong]["prompt"])
        multi = any(f.startswith("multi_instance:") for f in pa["flags"])
        if set(pa["objects"]) != set(pb["objects"]):
            lab, why = "물체 구성이 다른 후보", ""
        else:
            diffs = compare(pa, pb)
            obj = sorted({d[0] for d in diffs if d[1] == "explicit" and d[2] != "scene"})
            if multi:
                lab = "같은 종류 물체가 여러 개"
            elif not obj:
                lab = "명시 차이 없음(장면만 다르거나 파서가 놓침)"
            else:
                lab = "+".join(CAT_KO.get(x, x) for x in obj)
            why = fmt_diffs(diffs, "explicit")
        labels[lab] += 1
        details.append((gold, wrong, lab, why, by_id[gold]["prompt"]))
    L.append(f"\n===== 최종 2위로 남은 {len(details)}개: 정답과 1위 오답의 차이 =====")
    for lab, cnt in labels.most_common():
        L.append(f"  {cnt:3d}  {lab}")
    L.append(f"\n----- 사례 {min(n_cases, len(details))}개 (seed {seed}) -----")
    for gold, wrong, lab, why, text in random.Random(seed).sample(details, min(n_cases, len(details))):
        L.append(f"  [{gold} 정답 / {wrong} 1위] {lab} | {why[:90]}")
        L.append(f"      {text[:160]}")
    return "\n".join(L)
