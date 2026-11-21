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
