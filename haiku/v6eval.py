"""v6 후보 평가: 셀 5가 저장한 v3 재현 결과(질의별 후보 순서·점수)를 다시 써서 BLIP 재학습 없이 빠르게 비교한다.

비교 (모두 상위 2장의 순서만 바꿈, 기준 = 제출한 v4):
  v4            : OWL 좌우·상하, 기준 0.1, 기존 규칙 (2위 관계 점수 > 1위면 교환)
  엄격          : 두 이미지 모두에서 물체를 확신 있게 찾았고, 1위는 관계를 어기고 2위는 지킬 때만 교환
  C             : 엄격 + (OWL 직전 1·2위 점수 차가 작음) + (2위의 조건 점수가 1위보다 뚜렷하게 낮지 않음)
  + 개체 구별   : 같은 종류 물체가 여러 개인 질의에서, 상자를 여러 개 찾아 개체-속성 결합이 맞는지 확인.
                  1위는 결합이 틀리고(여백 < 0) 2위는 맞을 때(여백 > 0)만 교환
선택 규칙: 개발·검증 둘 다 v4보다 (고침 − 망침) > 0, 교환 정확도(고침 ÷ (고침+망침)) ≥ 0.7.
값(박빙 기준, 조건 차이 기준)은 개발에서만 정하고 검증은 보고만 한다.
"""

import math

import torch

from haiku.owl import BASE_RELS, should_swap, spatial_relations
from haiku.siblings import parse_prompt


def _score(rk):
    return sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)


def evaluate(rows, saved_path, owl_chk, binder=None, gap_q=(0.25, 0.5, 0.75, 1.01), cond_q=(0.0, 0.1, 0.25, 1.01)):
    S = torch.load(saved_path, weights_only=False)
    by_id = {r["id"]: r for r in rows}
    splits = {}
    for name in ("dev", "val"):
        g_ids = S["dev"] if name == "dev" else S["va"]
        items = S[f"{name}_items"]
        orders = S["v3_orders"][name]
        splits[name] = {"g": g_ids, "items": items, "orders": [(list(o), list(sc)) for o, sc in orders]}

    rank = lambda o, it: o.index(it["gold"]) + 1 if it["gold"] in o else it["base_rank"]
    path = lambda sp, j: by_id[sp["g"][j]]["image_path"]
    prompt = lambda it: by_id[it["q"]]["prompt"]

    # OWL 검출 (셀 5와 같은 OwlChecker를 넘기면 캐시를 그대로 씀)
    for sp in splits.values():
        prep, need = [], []
        for it, (o, sc) in zip(sp["items"], sp["orders"]):
            rels = spatial_relations(prompt(it), BASE_RELS)
            if rels and len(o) >= 2:
                names = tuple(sorted({x for _, a, b in rels for x in (a, b)}))
                paths = [path(sp, o[0]), path(sp, o[1])]
                need += [(p, names) for p in paths]
                prep.append((rels, names, paths))
            else:
                prep.append(None)
        owl_chk.detect(need)
        sp["prep"] = prep
        sB = [dict(zip(it["idxB"], it["sB"])) for it in sp["items"]]
        sp["gap"] = [sc[0] - sc[1] if len(sc) > 1 else 9.0 for _, sc in sp["orders"]]
        sp["cond_drop"] = [sB[n][o[0]] - sB[n][o[1]] for n, (o, _) in enumerate(sp["orders"])]

    def owl_orders(sp, strict, gap_max=None, drop_max=None):
        out = []
        for n, ((o, sc), pr) in enumerate(zip(sp["orders"], sp["prep"])):
            if pr and (gap_max is None or sp["gap"][n] <= gap_max) and (drop_max is None or sp["cond_drop"][n] <= drop_max):
                rels, names, paths = pr
                if should_swap(owl_chk.cache[(paths[0], names)], owl_chk.cache[(paths[1], names)], rels, 0.1, 0.06, strict):
                    o, sc = [o[1], o[0]] + o[2:], [sc[1], sc[0]] + sc[2:]
            out.append((o, sc))
        return out

    def stats(sp, orders, base_rk):
        rk = [rank(o, it) for (o, _), it in zip(orders, sp["items"])]
        fx = sum(b > 1 and r == 1 for b, r in zip(base_rk, rk))
        br = sum(b == 1 and r > 1 for b, r in zip(base_rk, rk))
        both = sum(o1[0] != o2[0] and b > 1 and r > 1 for (o1, _), (o2, _), b, r
                   in zip(orders, sp["v4_orders"], base_rk, rk))
        return {"score": round(_score(rk), 4), "R@1": round(sum(r == 1 for r in rk) / len(rk), 4),
                "fixed": fx, "broken": br, "swap_both_wrong": both, "precision": round(fx / max(fx + br, 1), 3)}, rk

    for sp in splits.values():
        sp["v4_orders"] = owl_orders(sp, False)
        sp["v4_rk"] = [rank(o, it) for (o, _), it in zip(sp["v4_orders"], sp["items"])]

    res = {"variants": {}}
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * (len(xs) - 1)))] if p <= 1 else float("inf")
    cand = {"v4": dict(strict=False), "엄격": dict(strict=True)}
    dev = splits["dev"]
    best, best_net = None, -1e9                                        # C: 개발에서 박빙·조건 기준 선택
    for gq in gap_q:
        for cq in cond_q:
            gmax = q(dev["gap"], gq)
            dmax = q([d for d in dev["cond_drop"] if d > 0] or [0.0], cq) if cq <= 1 else None
            st, _ = stats(dev, owl_orders(dev, True, gmax, dmax), dev["v4_rk"])
            net = st["fixed"] - st["broken"]
            if net > best_net:
                best, best_net = (gq, cq), net
    gq, cq = best
    gmax_c = q(dev["gap"], gq)
    dmax_c = q([d for d in dev["cond_drop"] if d > 0] or [0.0], cq) if cq <= 1 else None
    cand["C (엄격+박빙+조건)"] = dict(strict=True, gmax=gmax_c, dmax=dmax_c)
    res["C_chosen_on_dev"] = {"gap_quantile": gq, "gap_max": round(gmax_c, 4) if gmax_c != float("inf") else None,
                              "cond_quantile": cq, "cond_drop_max": None if dmax_c is None else round(dmax_c, 4)}

    final = {}
    for name, cfg in cand.items():
        row = {}
        for sname, sp in splits.items():
            orders = owl_orders(sp, cfg["strict"], cfg.get("gmax"), cfg.get("dmax"))
            row[sname], _ = stats(sp, orders, sp["v4_rk"])
            final[(name, sname)] = orders
        res["variants"][name] = row

    if binder is not None:                                              # 개체 구별: v4 순서와 C 순서 위에 각각
        for base_name in ("v4", "C (엄격+박빙+조건)"):
            margins = {}
            for sname, sp in splits.items():
                orders = final[(base_name, sname)]
                its = [(path(sp, o[j]), prompt(it)) for (o, _), it in zip(orders, sp["items"]) for j in (0, 1)]
                margins[sname] = binder.margins(its)

            def apply(sname, rule, delta=0.0):
                sp, m = splits[sname], margins[sname]
                new, decided = [], 0
                for n, (o, sc) in enumerate(final[(base_name, sname)]):
                    ma, mb = m[2 * n], m[2 * n + 1]
                    if ma is not None and mb is not None:
                        decided += 1
                        if (rule == "sign" and ma < 0 < mb) or (rule == "rel" and mb - ma > delta):
                            o, sc = [o[1], o[0]] + o[2:], [sc[1], sc[0]] + sc[2:]
                    new.append((o, sc))
                return new, decided

            dev_nets = {}
            for dlt in (0.01, 0.02, 0.03, 0.05, 0.08):
                st, _ = stats(splits["dev"], apply("dev", "rel", dlt)[0], splits["dev"]["v4_rk"])
                dev_nets[dlt] = st["fixed"] - st["broken"]
            dlt = max(dev_nets, key=dev_nets.get)
            for rule, label in (("sign", "개체 구별(부호)"), ("rel", f"개체 구별(차이>{dlt})")):
                row = {}
                for sname, sp in splits.items():
                    new, decided = apply(sname, rule, dlt)
                    st, rk = stats(sp, new, sp["v4_rk"])
                    st["decided"] = decided
                    multi = [n for n, it in enumerate(sp["items"])
                             if any(f.startswith("multi_instance:") for f in parse_prompt(prompt(it))["flags"])]
                    st["multi_R@1"] = (round(sum(sp["v4_rk"][n] == 1 for n in multi) / max(len(multi), 1), 3),
                                       round(sum(rk[n] == 1 for n in multi) / max(len(multi), 1), 3), len(multi))
                    row[sname] = st
                res["variants"][f"{base_name} + {label}"] = row

    ok = lambda r: all(r[s]["fixed"] - r[s]["broken"] > 0 and r[s]["precision"] >= 0.7 for s in ("dev", "val"))
    passed = [k for k, r in res["variants"].items() if k != "v4" and ok(r)]
    res["passed"] = passed
    res["pick"] = max(passed, key=lambda k: res["variants"][k]["val"]["fixed"] - res["variants"][k]["val"]["broken"]) if passed else "v4"
    return res


def report(res):
    L = ["===== v6 후보 비교 (기준 = v4, 상위 2장 교환만) =====",
         f"  {'방식':24s} | {'개발 점수':>8s} 고침/망침 정확도 | {'검증 점수':>8s} 고침/망침 정확도 둘다오답교환"]
    for name, r in res["variants"].items():
        d, v = r["dev"], r["val"]
        extra = ""
        if "decided" in v:
            extra = f" | 판정 가능 {v['decided']}질의, 같은 종류 여럿 R@1 {v['multi_R@1'][0]}→{v['multi_R@1'][1]} (n={v['multi_R@1'][2]})"
        L.append(f"  {name:24s} | {d['score']:8.4f} {d['fixed']:3d}/{d['broken']:<3d} {d['precision']:5.2f} | "
                 f"{v['score']:8.4f} {v['fixed']:3d}/{v['broken']:<3d} {v['precision']:5.2f} {v['swap_both_wrong']:5d}{extra}")
    L.append(f"  C 기준(개발에서 선택): {res['C_chosen_on_dev']}")
    L.append(f"  선택 규칙 통과: {res['passed'] or '없음'} → v6 추천: {res['pick']}")
    return "\n".join(L)
