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
        g_ids = S.get("g_dev", S["dev"]) if name == "dev" else S.get("g_val", S["va"])
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


def k_breakdown(saved_path, base_k=5, ks=(7, 10)):
    """저장 결과만으로 (GPU 없이) BLIP 후보 수 k의 효과를 질의 단위로 센다 (OWL 전, v3 단계).
    - 기준 k에서 'BLIP이 못 본 정답' = 조건 재정렬 후 정답이 k+1위 이하이지만 검색 후보(K장) 안에 있는 질의
    - 새 k에서 그중 몇 개가 5위 안 / 1위로 돌아왔는지, 기준 k에서 1위였던 질의 중 몇 개를 잃었는지."""
    S = torch.load(saved_path, weights_only=False)
    curve = S.get("k_curve", {})
    items = S["val_items"]
    rank = lambda o, it: o.index(it["gold"]) + 1 if it["gold"] in o else it["base_rank"]

    def ranks(k):
        lam = curve.get(k, {}).get("lam", S["lam"])
        out = []
        for it in items:
            sB = torch.tensor(it["sB"][:k]) + lam * torch.tensor(it["itm"][:k])
            o = [it["idxB"][j] for j in sB.argsort(descending=True).tolist()] + it["idxB"][k:]
            out.append(rank(o, it))
        return out

    cond = [it["idxB"].index(it["gold"]) + 1 if it["gold"] in it["idxB"] else it["base_rank"] for it in items]
    K = len(items[0]["idxB"])
    base = ranks(base_k)
    unseen = [n for n, c in enumerate(cond) if base_k < c <= K]
    L = [f"===== BLIP 후보 수 k 효과 (검증 {len(items)}질의, OWL 전 단계, 기준 k={base_k}) =====",
         f"  검색 후보 {K}장 밖에 정답: {sum(c > K for c in cond)} | 조건 재정렬 후 {base_k + 1}~{K}위라 BLIP이 못 본 정답: {len(unseen)} "
         f"(6~7위 {sum(c <= 7 for c in (cond[n] for n in unseen))}, 8~{K}위 {sum(c > 7 for c in (cond[n] for n in unseen))})",
         f"  k={base_k}: 점수 {_score(base):.4f}  1위 {sum(r == 1 for r in base)}  5위 안 {sum(r <= 5 for r in base)}"]
    for k in ks:
        if k > len(items[0]["itm"]):
            continue
        rk = ranks(k)
        rec5 = sum(rk[n] <= 5 for n in unseen)
        rec1 = sum(rk[n] == 1 for n in unseen)
        lost = sum(b == 1 and r > 1 for b, r in zip(base, rk))
        gain = sum(b > 1 and r == 1 for b, r in zip(base, rk))
        L.append(f"  k={k:2d}: 점수 {_score(rk):.4f}  1위 {sum(r == 1 for r in rk)}  5위 안 {sum(r <= 5 for r in rk)} | "
                 f"못 보던 {len(unseen)}개 중 5위 안 복구 {rec5}, 1위 복구 {rec1} | 1위 새로 맞힘 {gain}, 기존 1위 잃음 {lost}")
    return "\n".join(L)


def mix_breakdown(saved_path, base_k=5):
    """BLIP에 넘길 후보를 '조건 재정렬 상위 + 아직 안 들어간 SigLIP 상위'로 섞는 방식 비교 (저장 결과만, GPU 없음).
      A: 조건 상위 5 | B: 조건 상위 4 + SigLIP 1 (BLIP 호출 수 같음) | C: 조건 상위 5 + SigLIP 2 (7장)
      D: 조건 상위 7 | E: 조건 상위 10
    BLIP 점수가 있는 후보(조건 순위 상위 k_max) 안에서만 섞는다. 나머지 순서는 조건 재정렬 순서를 따른다."""
    S = torch.load(saved_path, weights_only=False)
    items = S["val_items"]
    if "idxA" not in items[0]:
        return "(이 저장 파일에는 SigLIP 순서가 없어 혼합 비교를 할 수 없습니다 — 새 코드로 다시 저장 필요)"
    curve = S.get("k_curve", {})
    lam_of = lambda n: curve.get(n, curve.get(min(curve, key=lambda k: abs(k - n)), {})).get("lam", S["lam"]) if curve else S["lam"]
    rank = lambda o, it: o.index(it["gold"]) + 1 if it["gold"] in o else it["base_rank"]
    kmax = len(items[0]["itm"])

    def order_for(it, n_cond, n_sig):
        sel = list(it["idxB"][:n_cond])
        for j in it["idxA"]:
            if len([x for x in sel if x not in it["idxB"][:n_cond]]) >= n_sig:
                break
            if j not in sel and j in it["idxB"][:kmax]:
                sel.append(j)
        pos = {j: p for p, j in enumerate(it["idxB"])}
        lam = lam_of(len(sel))
        sc = {j: it["sB"][pos[j]] + lam * it["itm"][pos[j]] for j in sel}
        head = sorted(sel, key=lambda j: -sc[j])
        return head + [j for j in it["idxB"] if j not in sel]

    plans = {"A 조건 5": (5, 0), "B 조건 4 + SigLIP 1": (4, 1), "C 조건 5 + SigLIP 2": (5, 2), "D 조건 7": (7, 0), "E 조건 10": (10, 0)}
    cond = [it["idxB"].index(it["gold"]) + 1 if it["gold"] in it["idxB"] else it["base_rank"] for it in items]
    K = len(items[0]["idxB"])
    unseen = [n for n, c in enumerate(cond) if base_k < c <= K]
    sig5 = sum(items[n]["base_rank"] <= 5 for n in unseen)
    L = [f"===== BLIP 후보 혼합 비교 (검증 {len(items)}질의, OWL 전 단계) =====",
         f"  조건 재정렬 후 BLIP 밖(6~{K}위)으로 밀린 정답 {len(unseen)}개 중 SigLIP 순위로는 5위 안이었던 것: {sig5}개"]
    base = [rank(order_for(it, 5, 0), it) for it in items]
    for name, (nc, ns) in plans.items():
        if nc > kmax:
            continue
        rk = [rank(order_for(it, nc, ns), it) for it in items]
        rec5 = sum(rk[n] <= 5 for n in unseen)
        rec1 = sum(rk[n] == 1 for n in unseen)
        lost = sum(b == 1 and r > 1 for b, r in zip(base, rk))
        gain = sum(b > 1 and r == 1 for b, r in zip(base, rk))
        L.append(f"  {name:20s} BLIP {nc + ns:2d}장 | 점수 {_score(rk):.4f}  1위 {sum(r == 1 for r in rk)}  5위 안 {sum(r <= 5 for r in rk)} | "
                 f"밀린 {len(unseen)}개 중 5위 안 {rec5}·1위 {rec1} | A 대비 1위 새로 {gain}, 잃음 {lost}")
    return "\n".join(L)


def plural_eval(rows, saved_path, owl_chk, conf_grid=(0.05, 0.1, 0.15, 0.2), margin=0.08, max_boxes=4):
    """복수 주체 관계(두 X가 한 Y의 위·아래·왼쪽·오른쪽) 점검을 v4 순서 위에 더했을 때 (OWL 여러 상자 검출, 팀원 규칙).
    확신도 기준은 개발에서 고르고 검증은 보고만 한다. 1위 contradict + 2위 satisfy일 때만 교환."""
    from haiku.plural import owl_found, parse_plural_plan, plural_geometry
    S = torch.load(saved_path, weights_only=False)
    by_id = {r["id"]: r for r in rows}
    rank = lambda o, it: o.index(it["gold"]) + 1 if it["gold"] in o else it["base_rank"]
    out = {}
    per_split = {}
    for name in ("dev", "val"):
        g = S.get("g_dev", S["dev"]) if name == "dev" else S.get("g_val", S["va"])
        items, orders = S[f"{name}_items"], [(list(o), list(sc)) for o, sc in S["v3_orders"][name]]
        prompt = lambda it: by_id[it["q"]]["prompt"]
        path = (lambda gg: (lambda j: by_id[gg[j]]["image_path"]))(g)     # 분할마다 갤러리를 고정 (늦은 바인딩 방지)
        # v4 순서 (OWL 좌우·상하, 0.1/0.06)
        prep, need = [], []
        for it, (o, _) in zip(items, orders):
            rels = spatial_relations(prompt(it), BASE_RELS)
            if rels and len(o) >= 2:
                names = tuple(sorted({x for _, a, b in rels for x in (a, b)}))
                need += [(path(o[0]), names), (path(o[1]), names)]
                prep.append((rels, names))
            else:
                prep.append(None)
        owl_chk.detect(need)
        v4 = []
        for (o, sc), pr in zip(orders, prep):
            if pr and should_swap(owl_chk.cache[(path(o[0]), pr[1])], owl_chk.cache[(path(o[1]), pr[1])], pr[0], 0.1, 0.06):
                o = [o[1], o[0]] + o[2:]
            v4.append(o)
        plans = {n: parse_plural_plan(prompt(it)) for n, it in enumerate(items)}
        plans = {n: p["relations"][0] for n, p in plans.items() if p["eligible"]}
        jobs = [(path(v4[n][j]), lab) for n, p in plans.items() for j in (0, 1) for lab in (p["subject"], p["object"])]
        owl_chk.detect_multi(list(dict.fromkeys(jobs)), max_boxes, 0.03)
        per_split[name] = (items, v4, plans, path)
    for conf in conf_grid:
        for name, (items, v4, plans, path) in per_split.items():
            base_rk = [rank(o, it) for o, it in zip(v4, items)]
            rk, decided, swapped = list(base_rk), 0, 0
            for n, p in plans.items():
                o = v4[n]
                verdicts = []
                for j in (0, 1):
                    found = {p["subject"]: owl_found(owl_chk.multi_cache.get((path(o[j]), p["subject"]), [])),
                             p["object"]: owl_found(owl_chk.multi_cache.get((path(o[j]), p["object"]), []))}
                    verdicts.append(plural_geometry(found, p["subject"], p["relation"], p["object"], margin=margin, min_conf=conf)["verdict"])
                decided += all(v != "abstain" for v in verdicts)
                if verdicts == ["contradict", "satisfy"]:
                    swapped += 1
                    rk[n] = rank([o[1], o[0]] + o[2:], items[n])
            fx = sum(b > 1 and r == 1 for b, r in zip(base_rk, rk))
            br = sum(b == 1 and r > 1 for b, r in zip(base_rk, rk))
            out[(conf, name)] = {"targets": len(plans), "decided_both": decided, "swapped": swapped, "fixed": fx, "broken": br,
                                 "neutral": swapped - fx - br, "score_v4": round(_score(base_rk), 4), "score": round(_score(rk), 4)}
    best = max(conf_grid, key=lambda c: (out[(c, "dev")]["fixed"] - out[(c, "dev")]["broken"], c))
    v, d = out[(best, "val")], out[(best, "dev")]
    tot_fx, tot_br = v["fixed"] + d["fixed"], v["broken"] + d["broken"]
    passed = d["fixed"] - d["broken"] >= 0 and v["fixed"] - v["broken"] > 0 and tot_fx / max(tot_fx + tot_br, 1) >= 0.75
    return {"table": out, "conf": best, "passed": passed}


def plural_report(res):
    L = ["===== 복수 주체 관계 점검 (두 X가 한 Y 기준, OWL 여러 상자, 기준 = v4 순서) ====="]
    for (conf, name), r in sorted(res["table"].items()):
        L.append(f"  확신도 {conf:.2f} {name:3s} | 대상 {r['targets']:3d} 둘 다 판정 {r['decided_both']:3d} 교환 {r['swapped']:2d} | "
                 f"고침 {r['fixed']} 망침 {r['broken']} 중립 {r['neutral']} | 점수 {r['score_v4']:.4f} → {r['score']:.4f}")
    L.append(f"  개발에서 고른 확신도 {res['conf']} → 채택 {'✅' if res['passed'] else '❌'} "
             "(개발 순이익 ≥ 0, 검증 순이익 > 0, 교환 정확도 ≥ 75%)")
    return "\n".join(L)
