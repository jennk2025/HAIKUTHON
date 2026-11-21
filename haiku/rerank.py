"""조건 분해 재정렬: 질의 하나를 객체별 조건으로 나누고, 현재 후보(Top-K)를 잘 가르는 조건에 비중을 준다.

- 조건 추출: haiku.siblings.parse_prompt (규칙 기반). 객체별 색·무늬·재질·모양·상태·자세·개수, 두 객체의 색/개수 결합,
  관계(주체·방향·대상)를 보존한다. 같은 종류 물체가 여러 개라 대상이 모호하면 그 조건은 만들지 않는다.
- 대조 문구: 현재 질의 문장에서 규칙으로 만든다 (다른 값, 반대 상태, 뒤집힌 관계, 맞바꾼 결합). 점검·추론 전용.
- 조건 점수 m_c(j) = sim(목표 문구, 후보 j) - max sim(대조 문구, 후보 j)
- 방식: A 원래 순위 / B 조건을 같은 비중으로 / C 개발 판별력 × 현재 후보 간 차이로 비중 / D 1·2위가 박빙인 질의에만 C
  B와 C는 조건 비중의 합이 1로 같고, 원래 점수에 lam × (가중 조건 점수)를 더한다. 조건 하나로 후보를 탈락시키지 않는다.
- 순위는 현재 질의 문장과 그 질의의 후보 이미지 임베딩만으로 계산한다.
  다른 질의의 결과, 후보 이미지의 원래 설명, 그룹 ID, 정답 ID는 순위 계산에 쓰지 않는다.
"""

import csv
import math
import time
from itertools import combinations

import torch

from haiku.siblings import _np, parse_prompt

VOCAB = {
    "color": "beige black blue brown gray green orange pink purple red white yellow".split(),
    "pattern": "solid-colored striped polka-dotted checkered floral".split(),
    "material": "ceramic fabric glass leather metal plastic rubber stone wicker wooden".split(),
    "shape": "hexagonal oval rectangular round square star-shaped triangular".split(),
    "pose": "lying sitting standing kneeling crouching".split(),
}
STATE_OPPOSITE = {"open": ["closed"], "closed": ["open"], "folded": ["unfolded"], "unfolded": ["folded"],
                  "lit": ["unlit"], "unlit": ["lit"]}
# 2-1 점검 범주 → 재정렬 조건 종류
PROBE_TO_KIND = {"color": "color", "pattern": "pattern", "material": "material", "shape": "shape", "state": "state",
                 "pose": "pose", "count_binding_swap": "count_binding", "color_binding_swap": "color_binding",
                 "relation_left_right": "relation_left_right", "relation_above_below": "relation_above_below"}


# ----------------------------------------------------------------------------- 조건 추출
def extract_conditions(prompt):
    """질의 문장 하나 → 조건 목록 [{kind, key, target, contrasts}]. 다른 질의나 후보 정보는 쓰지 않는다."""
    p = parse_prompt(prompt)
    multi = {f.split(":", 1)[1] for f in p["flags"] if f.startswith("multi_instance:")}
    objs = {n: o for n, o in p["objects"].items() if n not in multi}
    conds = []
    for name, o in objs.items():
        cnt = o["count"]
        for cat in ("color", "pattern", "material", "shape", "state", "pose"):
            vals = o["attrs"].get(cat, [])
            if len(vals) != 1:
                continue
            v = vals[0]
            others = STATE_OPPOSITE.get(v, []) if cat == "state" else [x for x in VOCAB[cat] if x != v]
            if others:
                conds.append({"kind": cat, "key": f"{cat}[{name}]={v}", "target": _np(name, cnt, v),
                              "contrasts": [_np(name, cnt, x) for x in others]})
        if cnt in (1, 2, 3):
            conds.append({"kind": "count", "key": f"count[{name}]={cnt}", "target": _np(name, cnt),
                          "contrasts": [_np(name, c) for c in (1, 2, 3) if c != cnt]})
    colored = [n for n in objs if len(objs[n]["attrs"].get("color", [])) == 1]
    for a, b in combinations(sorted(colored), 2):
        ca, cb = objs[a]["attrs"]["color"][0], objs[b]["attrs"]["color"][0]
        if ca != cb:
            na, nb = objs[a]["count"], objs[b]["count"]
            conds.append({"kind": "color_binding", "key": f"color_binding[{a}={ca},{b}={cb}]",
                          "target": f"{_np(a, na, ca)} and {_np(b, nb, cb)}",
                          "contrasts": [f"{_np(a, na, cb)} and {_np(b, nb, ca)}"]})
    for a, b in combinations(sorted(objs), 2):
        na, nb = objs[a]["count"], objs[b]["count"]
        if na != nb and na in (1, 2, 3) and nb in (1, 2, 3):
            conds.append({"kind": "count_binding", "key": f"count_binding[{a}={na},{b}={nb}]",
                          "target": f"{_np(a, na)} and {_np(b, nb)}", "contrasts": [f"{_np(a, nb)} and {_np(b, na)}"]})
    for rel, x, y in p["relations"]:
        if x not in objs or y not in objs:
            continue
        nx, ny = _np(x, objs[x]["count"]), _np(y, objs[y]["count"])
        if rel == "left_of":
            conds.append({"kind": "relation_left_right", "key": f"{x} left_of {y}",
                          "target": f"{nx} to the left of {ny}", "contrasts": [f"{nx} to the right of {ny}"]})
        elif rel == "above":
            conds.append({"kind": "relation_above_below", "key": f"{x} above {y}",
                          "target": f"{nx} above {ny}", "contrasts": [f"{nx} below {ny}"]})
        elif rel in ("on", "inside"):
            word = "on" if rel == "on" else "inside"
            conds.append({"kind": "relation_on_inside", "key": f"{x} {rel} {y}",
                          "target": f"{nx} {word} {ny}", "contrasts": [f"{ny} {word} {nx}"]})
    return conds


# ----------------------------------------------------------------------------- 문구 임베딩 (같은 문구는 한 번만 계산)
class PhraseBank:
    """같은 문구의 임베딩을 재사용한다. 값은 문구만으로 정해지므로 다른 질의의 결과에 의존하지 않는다."""

    def __init__(self, encode_texts):
        self.encode = encode_texts
        self.index, self.chunks, self._mat = {}, [], None
        self.encode_sec = 0.0

    def add(self, phrases):
        new = [p for p in dict.fromkeys(phrases) if p not in self.index]
        if not new:
            return
        t0 = time.perf_counter()
        vecs = self.encode(new)
        self.encode_sec += time.perf_counter() - t0
        base = len(self.index)
        for k, p in enumerate(new):
            self.index[p] = base + k
        self.chunks.append(vecs)
        self._mat = None

    def mat(self, phrases):
        if self._mat is None:
            self._mat = torch.cat(self.chunks)
        return self._mat[[self.index[p] for p in phrases]]


def all_phrases(conds):
    return [p for c in conds for p in [c["target"], *c["contrasts"]]]


def condition_margins(conds, bank, img_k):
    """조건별 후보 점수 m_c(j): (조건 수, K)."""
    if not conds:
        return torch.zeros(0, img_k.shape[0], device=img_k.device)
    rows = []
    for c in conds:
        t = bank.mat([c["target"]]).to(img_k) @ img_k.T
        cs = bank.mat(c["contrasts"]).to(img_k) @ img_k.T
        rows.append((t - cs.max(0).values).squeeze(0))
    return torch.stack(rows)


def condition_weights(conds, M, method, reliability):
    n = len(conds)
    if n == 0:
        return None
    if method == "B":
        return torch.full((n,), 1.0 / n, device=M.device)
    r = torch.tensor([reliability.get(c["kind"], 0.0) for c in conds], device=M.device)
    w = r * M.std(dim=1, unbiased=False)                   # 개발 판별력 × 현재 후보 사이의 차이
    if float(w.sum()) <= 0:
        return None
    return w / w.sum()


def rerank_scores(s_k, conds, M, method, lam, reliability, tau=None):
    """후보 K개의 새 점수. 원래 점수에 조건 점수를 더하기만 한다."""
    if method == "A" or not conds:
        return s_k
    if method == "D":
        if len(s_k) >= 2:
            top2 = s_k.topk(2).values
            if float(top2[0] - top2[1]) >= tau:
                return s_k
        method = "C"
    w = condition_weights(conds, M, method, reliability)
    if w is None:
        return s_k
    return s_k + lam * (w @ M)


# ----------------------------------------------------------------------------- 2-1 속성 판별 점검
def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def probe(contrast_rows, img_vec, bank):
    """대조 쌍마다 문구 A·B × 이미지 A·B 점수 4개. 문구 A는 이미지 A를, 문구 B는 이미지 B를 더 높게 봐야 성공."""
    stats = {}
    for r in contrast_rows:
        a, b = r["id_a"], r["id_b"]                         # 목표 문구는 항상 id_a 이미지에 맞는다
        T = bank.mat([r["target"], r["contrast"]])
        I = torch.stack([img_vec(a), img_vec(b)]).to(T)
        s = T @ I.T                                          # [[tA·iA, tA·iB], [tB·iA, tB·iB]]
        ok_a, ok_b = bool(s[0, 0] > s[0, 1]), bool(s[1, 1] > s[1, 0])
        st = stats.setdefault(r["category"], {"pairs": 0, "both": 0, "directions": 0, "same_image": 0})
        st["pairs"] += 1
        st["both"] += ok_a and ok_b
        st["directions"] += ok_a + ok_b
        st["same_image"] += ok_a != ok_b                      # 두 문구가 같은 이미지를 고름
    reliability = {}
    for cat, st in stats.items():
        n = 2 * st["pairs"]
        st["direction_acc"] = round(st["directions"] / n, 4)
        st["direction_acc_lower95"] = round(wilson_lower(st["directions"], n), 4)
        kind = PROBE_TO_KIND.get(cat)
        if kind:
            reliability[kind] = round(max(0.0, (st["direction_acc_lower95"] - 0.5) / 0.5), 4)
    if "count_binding" in reliability and "count" not in reliability:
        reliability["count"] = reliability["count_binding"]  # 단독 개수 쌍이 없어 개수 결합 교차로 대신함
    return stats, reliability


# ----------------------------------------------------------------------------- 실험 (개발 = 학습 분할, 최종 = 검증 분할)
def _prepare(q_ids, g_ids, txt, img, pos, conds, bank, K):
    G = img[[pos[g] for g in g_ids]]
    gpos = {g: k for k, g in enumerate(g_ids)}
    items = []
    for q in q_ids:
        s = txt[pos[q]] @ G.T
        top = s.topk(min(K, len(g_ids)))
        gold = gpos[q]
        base_rank = int((s >= s[gold]).sum())
        M = condition_margins(conds[q], bank, G[top.indices])
        gap = float(top.values[0] - top.values[1]) if len(top.values) > 1 else 1.0
        items.append({"q": q, "idx": top.indices, "s": top.values, "M": M, "gold": gold, "base_rank": base_rank,
                      "gap": gap})
    return items


def _ranks(items, conds, method, lam, reliability, tau=None, return_top1=False):
    ranks, top1 = [], []
    for it in items:
        new = rerank_scores(it["s"], conds[it["q"]], it["M"], method, lam, reliability, tau)
        order = it["idx"][new.argsort(descending=True)]
        hit = (order == it["gold"]).nonzero()
        ranks.append(int(hit[0, 0]) + 1 if len(hit) else it["base_rank"])
        top1.append(int(order[0]))
    return (ranks, top1) if return_top1 else ranks


def _score(ranks):
    return sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in ranks) / len(ranks)


def run_experiment(rows, split, contrast_csv, encode_images, encode_texts, K_list=(10, 20),
                   lam_grid=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2), tau_quantiles=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    from haiku.metrics import metrics_from_ranks
    by_id = {r["id"]: r for r in rows}
    tr, va = list(split["train_ids"]), list(split["valid_ids"])
    ids = tr + va
    pos = {i: k for k, i in enumerate(ids)}
    t0 = time.perf_counter()
    img = encode_images([by_id[i]["image_path"] for i in ids])
    txt = encode_texts([by_id[i]["prompt"] for i in ids])
    out = {"embed_sec": round(time.perf_counter() - t0, 1)}
    bank = PhraseBank(encode_texts)

    # 2-1: 학습 분할 안 대조 쌍으로 속성별 판별력
    trs = set(tr)
    crow = [r for r in csv.DictReader(open(contrast_csv, encoding="utf-8-sig")) if r["id_a"] in trs and r["id_b"] in trs]
    bank.add([p for r in crow for p in (r["target"], r["contrast"])])
    stats, reliability = probe(crow, lambda i: img[pos[i]], bank)
    out["probe"], out["reliability"] = stats, reliability

    # 조건 추출 (질의 문장 하나씩)
    t0 = time.perf_counter()
    conds = {i: extract_conditions(by_id[i]["prompt"]) for i in ids}
    out["extract_sec_per_query_ms"] = round((time.perf_counter() - t0) / len(ids) * 1000, 3)
    n_c = [len(conds[i]) for i in ids]
    out["conditions_per_query"] = round(sum(n_c) / len(n_c), 2)
    out["queries_without_conditions"] = sum(n == 0 for n in n_c)
    kinds = {}
    for i in ids:
        for c in conds[i]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    out["condition_kinds"] = kinds
    enc0 = bank.encode_sec
    bank.add([p for i in ids for p in all_phrases(conds[i])])
    out["unique_phrases"] = len(bank.index)
    out["phrase_encode_sec"] = round(bank.encode_sec - enc0, 1)

    out["by_K"] = {}
    for K in K_list:
        dev = _prepare(tr, tr, txt, img, pos, conds, bank, K)
        t0 = time.perf_counter()
        val = _prepare(va, va, txt, img, pos, conds, bank, K)
        val_prep = time.perf_counter() - t0
        res = {"dev_hit@K": round(sum(it["base_rank"] <= K for it in dev) / len(dev), 4),
               "val_hit@K": round(sum(it["base_rank"] <= K for it in val) / len(val), 4),
               "val_condition_score_sec": round(val_prep, 2), "val_queries": len(va)}
        # 개발 데이터로 lam, tau 결정 (검증 데이터는 보지 않음)
        best = {}
        for m in ("B", "C"):
            sc = {lam: _score(_ranks(dev, conds, m, lam, reliability)) for lam in lam_grid}
            best[m] = max(sc, key=sc.get)
            res[f"dev_{m}_by_lam"] = {str(k): round(v, 4) for k, v in sc.items()}
        gaps = sorted(it["gap"] for it in dev)
        taus = sorted({gaps[min(len(gaps) - 1, int(q * len(gaps)))] for q in tau_quantiles})
        sc = {t: _score(_ranks(dev, conds, "D", best["C"], reliability, t)) for t in taus}
        best_tau = max(sc, key=sc.get)
        res["dev_A"] = round(_score(_ranks(dev, conds, "A", 0, reliability)), 4)
        res["dev_D_by_tau"] = {f"{k:.4f}": round(v, 4) for k, v in sc.items()}
        res["chosen"] = {"lam_B": best["B"], "lam_C": best["C"], "tau_D": best_tau}
        # 검증: 고정된 값으로 한 번만 평가
        base_ranks, base_top1 = _ranks(val, conds, "A", 0, reliability, return_top1=True)
        for m, lam, tau in (("A", 0, None), ("B", best["B"], None), ("C", best["C"], None), ("D", best["C"], best_tau)):
            t0 = time.perf_counter()
            rk, top1 = _ranks(val, conds, m, lam, reliability, tau, return_top1=True)
            sec = time.perf_counter() - t0
            met = metrics_from_ranks(rk)
            res[f"val_{m}"] = {"score": round(met["score"], 4), "recall@1": round(met["recall@1"], 4),
                               "hit@5": round(met["hit@5"], 4),
                               "wrong_to_right@1": sum(b > 1 and r == 1 for b, r in zip(base_ranks, rk)),
                               "right_to_wrong@1": sum(b == 1 and r > 1 for b, r in zip(base_ranks, rk)),
                               "rerank_sec_800q": round(sec, 2)}
        res["val_examples_C"] = _examples(val, conds, best["C"], reliability, va)
        out["by_K"][str(K)] = res
    return out


def _examples(items, conds, lam, reliability, q_ids, n=6):
    """C에서 1위가 바뀐 질의: 어떤 조건이 순위를 바꿨는지."""
    ex = []
    for it in items:
        cs = conds[it["q"]]
        new = rerank_scores(it["s"], cs, it["M"], "C", lam, reliability)
        o_old, o_new = int(it["s"].argmax()), int(new.argmax())
        if o_old == o_new:
            continue
        w = condition_weights(cs, it["M"], "C", reliability)
        contrib = (w * (it["M"][:, o_new] - it["M"][:, o_old])).tolist()
        k = max(range(len(cs)), key=lambda c: contrib[c])
        ex.append({"query": it["q"], "result": "개선" if int(it["idx"][o_new]) == it["gold"] else (
            "악화" if int(it["idx"][o_old]) == it["gold"] else "변경(둘 다 오답)"), "condition": cs[k]["key"],
                   "contribution": round(contrib[k], 4)})
    ex.sort(key=lambda e: {"개선": 0, "악화": 1}.get(e["result"], 2))
    return ex[:n] + [e for e in ex if e["result"] == "악화"][:n]
