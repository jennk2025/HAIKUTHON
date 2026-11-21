"""검증 800질의 사례 내보내기: 셀 5(또는 셀 7)가 저장한 v3 재현 결과를 재사용해 질의별 단계 순위·점수·OWL 근거를 CSV로,
최종 2위 사례와 같은 유형의 정답 사례는 1·2위 후보 이미지까지 ZIP 하나로 묶는다 (공개 학습 데이터의 검증 분할, 분석용).

- 단계: 검색(SigLIP) → 조건 재정렬 → BLIP 재정렬(v3) → OWL v4 설정(좌우·상하 0.1/0.06) → 최종(저장된 v5 순서: 확장 관계)
- OWL 검출 상자는 저장돼 있지 않아 같은 설정으로 다시 계산한다 (같은 OwlChecker 규칙).
- 원인 분류는 정답 문장과 1위 오답 이미지의 원래 문장을 규칙 파서로 비교한 분석용 표시이며, 순위 계산에는 쓰지 않는다.
"""

import csv
import io
import json
import random
import zipfile
from pathlib import Path

import torch
from PIL import Image

from haiku.owl import BASE_RELS, EXT_RELS, should_swap, spatial_relations
from haiku.instances import parse_instances
from haiku.plural import parse_plural_plan
from haiku.siblings import CAT_KO, compare, fmt_diffs, parse_prompt

FIELDS_MISSING = [
    "SigLIP 원점수 (저장 안 됨 — 조건 재정렬 후 점수 sB와 검색 순위만 있음)",
    "조건별 여백 (조건 재정렬 내부 값, 저장 안 됨)",
    "정답 이미지가 상위 2장 밖일 때 정답 이미지의 OWL 검출 (상위 2장만 검출)",
]


def _label(prompt_gold, prompt_wrong):
    pa, pb = parse_prompt(prompt_gold), parse_prompt(prompt_wrong)
    multi = any(f.startswith("multi_instance:") for f in pa["flags"])
    if set(pa["objects"]) != set(pb["objects"]):
        return "물체 구성이 다른 후보", ""
    diffs = compare(pa, pb)
    obj = sorted({d[0] for d in diffs if d[1] == "explicit" and d[2] != "scene"})
    if multi:
        lab = "같은 종류 물체가 여러 개"
    elif not obj:
        lab = "명시 차이 없음(장면만 다르거나 파서가 놓침)"
    else:
        lab = "+".join(CAT_KO.get(x, x) for x in obj)
    return lab, fmt_diffs(diffs, "explicit")


def k_compare(rows, saved_path, owl_chk, old_k=5, new_k=10):
    """BLIP 후보 k=5(v4) vs k=10(v6) 질의별 비교 (저장 결과 재사용, 둘 다 v4 OWL 교환 적용).
    최종 1위가 바뀐 질의를 '복구 / 밀어냄 / 둘 다 오답' × '새 1위가 기존 BLIP 창(5장) 밖에서 왔는지'로 나눈다."""
    S = torch.load(saved_path, weights_only=False)
    by_id = {r["id"]: r for r in rows}
    g = S.get("g_val", S["va"])
    items = S["val_items"]
    curve = S.get("k_curve") or {}
    new_k = min(new_k, len(items[0]["itm"]))                            # BLIP 점수가 저장된 후보 수까지만
    path = lambda j: by_id[g[j]]["image_path"]

    def orders(k):
        lam = curve.get(k, {}).get("lam", S["lam"])
        out = []
        for it in items:
            sc = torch.tensor(it["sB"][:k]) + lam * torch.tensor(it["itm"][:k])
            out.append([it["idxB"][j] for j in sc.argsort(descending=True).tolist()] + list(it["idxB"][k:]))
        need, prep = [], []
        for it, o in zip(items, out):
            rels = spatial_relations(by_id[it["q"]]["prompt"], BASE_RELS)
            if rels and len(o) >= 2:
                names = tuple(sorted({x for _, a, b in rels for x in (a, b)}))
                need += [(path(o[0]), names), (path(o[1]), names)]
                prep.append((rels, names))
            else:
                prep.append(None)
        owl_chk.detect(need)
        for n, (o, pr) in enumerate(zip(out, prep)):
            if pr and should_swap(owl_chk.cache[(path(o[0]), pr[1])], owl_chk.cache[(path(o[1]), pr[1])], pr[0], 0.1, 0.06):
                out[n] = [o[1], o[0]] + o[2:]
        return out, lam

    o_old, lam_old = orders(old_k)
    o_new, lam_new = orders(new_k)
    recs, counts = [], {}
    for n, it in enumerate(items):
        a, b = o_old[n], o_new[n]
        gold = it["gold"]
        rk = lambda o: o.index(gold) + 1 if gold in o else it["base_rank"]
        cond = {j: r + 1 for r, j in enumerate(it["idxB"])}
        outside = cond.get(b[0], 99) > old_k
        if a[0] == b[0]:
            cat = "1위 같음"
        elif b[0] == gold:
            cat = "복구: 새 1위가 정답"
        elif a[0] == gold:
            cat = "밀어냄: 기존 정답 1위를 잃음"
        else:
            cat = "둘 다 오답"
        if a[0] != b[0]:
            cat += " (새 1위가 기존 5장 밖)" if outside else " (기존 5장 안 순서 변화)"
        counts[cat] = counts.get(cat, 0) + 1
        lab, det_ = "", ""
        if cat.startswith("밀어냄"):
            lab, det_ = _label(by_id[g[gold]]["prompt"], by_id[g[b[0]]]["prompt"])
        elif cat.startswith(("복구", "둘 다")):
            lab, det_ = _label(by_id[g[gold]]["prompt"], by_id[g[(a if cat.startswith("복구") else b)[0]]]["prompt"])
        recs.append({"query_id": it["q"], "prompt": by_id[it["q"]]["prompt"], "gold_id": g[gold],
                     "cond_rank_gold": cond.get(gold, it["base_rank"]),
                     f"k{old_k}_top1_id": g[a[0]], f"k{old_k}_top2_id": g[a[1]], f"k{old_k}_rank_gold": rk(a),
                     f"k{new_k}_top1_id": g[b[0]], f"k{new_k}_top2_id": g[b[1]], f"k{new_k}_rank_gold": rk(b),
                     f"k{new_k}_top1_cond_rank": cond.get(b[0], ""), "category": cat, "diff_label": lab, "diff_detail": det_,
                     "images": ""})
    for r in recs:
        if r["category"] != "1위 같음":
            ids = list(dict.fromkeys([r["gold_id"], r[f"k{old_k}_top1_id"], r[f"k{new_k}_top1_id"]]))
            r["images"] = ";".join(f"images/{i}.jpg" for i in ids)
    types = {}
    for r in recs:
        if r["category"].startswith("밀어냄"):
            types[r["diff_label"]] = types.get(r["diff_label"], 0) + 1
    return {"recs": recs, "counts": counts, "lost_types": types, "lam": (lam_old, lam_new), "file": Path(saved_path).name,
            "n_gallery": len(g)}


def export(rows, saved_path, owl_chk, out_zip, n_compare=10, img_size=512, seed=0, k_saved=None):
    S = torch.load(saved_path, weights_only=False)
    by_id = {r["id"]: r for r in rows}
    g = S.get("g_val", S["va"])
    items = S["val_items"]
    v3_orders = [(list(o), list(sc)) for o, sc in S["v3_orders"]["val"]]
    v5_orders = [(list(o), list(sc)) for o, sc in S["v5_orders"]["val"]] if "v5_orders" in S else v3_orders
    lam = S.get("lam", 0.1)
    path = lambda j: by_id[g[j]]["image_path"]
    rank = lambda o, it: o.index(it["gold"]) + 1 if it["gold"] in o else it["base_rank"]

    # OWL 검출 (v4 설정: 상위 2장 기준, 최종: 저장된 v5 상위 2장 기준)
    def prep(orders, kinds):
        out, need = [], []
        for it, (o, _) in zip(items, orders):
            rels = spatial_relations(by_id[it["q"]]["prompt"], kinds)
            if rels and len(o) >= 2:
                names = tuple(sorted({x for _, a, b in rels for x in (a, b)}))
                need += [(path(o[0]), names), (path(o[1]), names)]
                out.append((rels, names))
            else:
                out.append(None)
        owl_chk.detect(need)
        return out
    p4 = prep(v3_orders, BASE_RELS)
    v4_orders = []
    for (o, sc), pr in zip(v3_orders, p4):
        if pr and should_swap(owl_chk.cache[(path(o[0]), pr[1])], owl_chk.cache[(path(o[1]), pr[1])], pr[0], 0.1, 0.06):
            o, sc = [o[1], o[0]] + o[2:], [sc[1], sc[0]] + sc[2:]
        v4_orders.append((o, sc))
    p5 = prep(v5_orders, EXT_RELS)

    recs = []
    for n, it in enumerate(items):
        q = it["q"]
        itm = dict(zip(it["idxB"], it["itm"]))
        cs = dict(zip(it["idxB"], it["sB"]))
        inst = parse_instances(by_id[q]["prompt"])
        pl = parse_plural_plan(by_id[q]["prompt"])
        o3, sc3 = v3_orders[n]
        o4, _ = v4_orders[n]
        o5, _ = v5_orders[n]
        cond = it["idxB"].index(it["gold"]) + 1 if it["gold"] in it["idxB"] else it["base_rank"]
        det = lambda j, pr: json.dumps({k: [round(x, 3) for x in v] for k, v in owl_chk.cache[(path(j), pr[1])].items()},
                                       ensure_ascii=False) if pr else ""
        r = {"query_id": q, "split": "val", "prompt": by_id[q]["prompt"], "gold_id": q,
             "search_rank_gold": it["base_rank"], "cond_rank_gold": cond,
             "blip_top1_id": g[o3[0]], "blip_top2_id": g[o3[1]],
             "blip_top1_score": round(sc3[0], 4), "blip_top2_score": round(sc3[1], 4),
             "blip_top1_itm": round(itm.get(o3[0], float("nan")), 4), "blip_top2_itm": round(itm.get(o3[1], float("nan")), 4),
             "blip_top1_cond_score": round(cs.get(o3[0], float("nan")), 4), "blip_top2_cond_score": round(cs.get(o3[1], float("nan")), 4),
             "blip_rank_gold": rank(o3, it), "blip_lambda": lam,
             "v4_top1_id": g[o4[0]], "v4_top2_id": g[o4[1]], "v4_rank_gold": rank(o4, it), "v4_swapped": o4[0] != o3[0],
             "final_top1_id": g[o5[0]], "final_top2_id": g[o5[1]], "final_rank_gold": rank(o5, it),
             "final_swapped": o5[0] != o3[0],
             "owl_relations_v4": json.dumps(p4[n][0], ensure_ascii=False) if p4[n] else "",
             "owl_det_v4_top1": det(o3[0], p4[n]), "owl_det_v4_top2": det(o3[1], p4[n]),
             "owl_relations_final": json.dumps(p5[n][0], ensure_ascii=False) if p5[n] else "",
             "owl_det_final_top1": det(o5[0], p5[n]), "owl_det_final_top2": det(o5[1], p5[n]),
             "multi_instance_parsed": json.dumps(inst, ensure_ascii=False) if inst else "",
             "plural_relation": json.dumps(pl["relations"][0], ensure_ascii=False) if pl["eligible"] else "",
             "case_group": "", "diff_label": "", "diff_detail": "", "images": ""}
        recs.append(r)

    # 최종 2위 사례: 1위 오답과의 차이 / 같은 유형인데 최종 1위를 맞힌 비교 사례: 1위(정답)와 2위의 차이
    labels = {}
    for r in recs:
        if r["final_rank_gold"] == 2:
            r["diff_label"], r["diff_detail"] = _label(r["prompt"], by_id[r["final_top1_id"]]["prompt"])
            r["case_group"] = "최종 2위"
            labels.setdefault(r["diff_label"], 0)
            labels[r["diff_label"]] += 1
    correct = [r for r in recs if r["final_rank_gold"] == 1]
    pool = {}
    for r in correct:
        lab, det_ = _label(r["prompt"], by_id[r["final_top2_id"]]["prompt"])
        if lab in labels:
            pool.setdefault(lab, []).append((r, lab, det_))
    rng = random.Random(seed)
    for lab, cands in pool.items():
        for r, lab_, det_ in rng.sample(cands, min(n_compare, len(cands))):
            r["case_group"], r["diff_label"], r["diff_detail"] = "비교(최종 1위 정답)", lab_, det_

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    img_ids = set()
    K = k_compare(rows, k_saved, owl_chk) if k_saved else None
    if K:
        for r in K["recs"]:
            if r["images"]:
                img_ids.update(x[len("images/"):-len(".jpg")] for x in r["images"].split(";"))
    for r in recs:
        if r["case_group"]:
            ids = [r["final_top1_id"], r["final_top2_id"]]
            r["images"] = ";".join(f"images/{i}.jpg" for i in ids)
            img_ids.update(ids)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
        z.writestr("cases.csv", "﻿" + buf.getvalue())
        if K:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(K["recs"][0]))
            w.writeheader()
            w.writerows(K["recs"])
            z.writestr("k5_vs_k10.csv", "﻿" + buf.getvalue())
        for i in sorted(img_ids):
            with Image.open(by_id[i]["image_path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((img_size, img_size))
                b = io.BytesIO()
                im.save(b, "JPEG", quality=90)
                z.writestr(f"images/{i}.jpg", b.getvalue())
        readme = ["# 검증 800질의 사례 내보내기",
                  f"- 원본 저장 파일: {Path(saved_path).name} (BLIP 후보 수 k={S.get('k')}, λ={lam})",
                  "- 단계: 검색 → 조건 재정렬 → BLIP(v3) → OWL v4 설정(좌우·상하 0.1/0.06) → 최종(저장된 v5 순서: 확장 관계)",
                  "- 이미지: 최종 2위 사례와 비교 사례의 최종 1·2위 후보 (images/<이미지 ID>.jpg, 긴 변 512px). 이미지 ID = 그 이미지의 원래 질의 ID",
                  "- 최종 2위 원인 분류 (개수): " + ", ".join(f"{k} {v}" for k, v in sorted(labels.items(), key=lambda x: -x[1])),
                  "- 비교 사례: 같은 유형인데 최종 1위를 맞힌 질의를 유형마다 최대 %d개 (seed %d)" % (n_compare, seed),
                  "- OWL 검출 형식: {물체: [점수, 중심x, 중심y]} 또는 [점수, cx, cy, w, h] (이미지 크기 기준 0~1). "
                  "owl_det_v4_* = BLIP 직후 1·2위(v4 교환 판단에 쓴 두 장), owl_det_final_* = 최종 1·2위",
                  "- multi_instance_parsed: 같은 종류 물체가 여러 개이고 개체마다 속성이 다른 문장의 개체 분해 (개체 구별 실험의 파서)",
                  "- plural_relation: '두 X가 한 Y 위/아래/왼쪽/오른쪽' 문장의 분해 (복수 관계 점검 파서)",
                  *([f"- k5_vs_k10.csv: BLIP 후보 5장(v4) vs 10장(v6), 둘 다 v4 OWL 교환 적용, 원본 {K['file']} (후보 {K['n_gallery']}장, "
                     f"λ {K['lam'][0]} / {K['lam'][1]}). 1위가 바뀐 질의는 정답·기존 1위·새 1위 이미지 포함",
                     "  - 분류 (개수): " + ", ".join(f"{k} {v}" for k, v in sorted(K["counts"].items(), key=lambda x: -x[1])),
                     "  - 밀어냄 사례의 차이 유형: " + (", ".join(f"{k} {v}" for k, v in sorted(K["lost_types"].items(), key=lambda x: -x[1])) or "없음")]
                    if K else []),
                  "- 저장돼 있지 않아 포함하지 못한 항목:"] + [f"  - {m}" for m in FIELDS_MISSING]
        z.writestr("README.md", "\n".join(readme) + "\n")
    return {"zip": str(out_zip), "rows": len(recs), "rank2": sum(r["case_group"] == "최종 2위" for r in recs),
            "compare": sum(r["case_group"].startswith("비교") for r in recs), "images": len(img_ids),
            "labels": labels, "missing": FIELDS_MISSING,
            "k_counts": K["counts"] if K else None, "k_lost_types": K["lost_types"] if K else None,
            "k_file": K["file"] if K else None}
