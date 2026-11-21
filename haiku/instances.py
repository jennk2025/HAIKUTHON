"""같은 종류 물체가 여러 개인 질의: 물체를 하나씩 기록하고(개체 단위 파싱), 이미지에서 그 물체 상자들을 찾아
각 상자가 어느 설명(예: '열린 베이지 봉투' / '닫힌 흰 봉투')에 맞는지 짝지어 속성 결합이 맞는지 본다.

- 규칙 파서(haiku.siblings)의 토큰화·속성 연결 규칙을 그대로 쓰되, 속성을 물체 종류가 아니라 '언급 하나'에 붙인다.
- "one X" 소개가 개체 하나, "the gray X" 같은 지칭은 속성이 겹치는 개체 하나에 연결한다. 애매하면 그 종류는 쓰지 않는다.
- 이미지 점수 = (설명대로 짝지은 점수) − (속성을 개체끼리 맞바꾼 설명 중 최고 점수). 한 이미지 안에서만 짝짓는다.
- 각 질의의 점수는 그 질의 문장과 그 후보 이미지만으로 계산한다.
"""

import itertools

import torch

from haiku.siblings import ANIMATE, _is_subject, _np, _sentences, _tokenize, _typed_tokens, parse_prompt

CAT_ORDER = ["size", "state", "shape", "color", "pattern", "material"]


def _mentions(prompt):
    out = []
    for sent in _sentences(_tokenize(prompt)):
        S = _typed_tokens(sent)
        noun_idx = [k for k, t in enumerate(S) if t["t"] == "NOUN"]
        ment = {}
        for k in noun_idx:
            j = k - 1
            while j >= 0 and (S[j]["t"] in ("ATTR", "PUNCT") or (S[j]["t"] == "W" and S[j]["w"] in ("other", "and"))):
                j -= 1
            intro = j >= 0 and S[j]["t"] == "COUNT" and not (j > 0 and S[j - 1]["t"] == "W" and S[j - 1]["w"] == "the")
            ment[k] = {"type": S[k]["type"], "intro": intro, "count": S[j]["n"] if intro else 0, "attrs": {}}
        for q, tok in enumerate(S):
            if tok["t"] != "ATTR":
                continue
            target = None
            j = q + 1
            while j < len(S) and (S[j]["t"] == "PUNCT" or (S[j]["t"] == "ATTR" and not S[j].get("verb"))
                                  or (S[j]["t"] == "W" and S[j]["w"] == "and")):
                j += 1
            if j < len(S) and S[j]["t"] == "NOUN" and not tok.get("verb"):
                target = j
            else:
                before = [k for k in noun_idx if k < q]
                if before:
                    nb = before[-1]
                    between = [x.get("w") for x in S[nb + 1:q]]
                    if "and" in between and ("is" in between or "are" in between):
                        subj = [k for k in noun_idx if k < q and _is_subject(S, k)]
                        nb = subj[-1] if subj else noun_idx[0]
                    target = nb
            if target is None or (tok["cat"] == "pose" and S[target]["type"] not in ANIMATE):
                continue
            ment[target]["attrs"].setdefault(tok["cat"], set()).add(tok["val"])
        out += [ment[k] for k in noun_idx]
    return out


def parse_instances(prompt, max_instances=3):
    """→ {물체 종류: {"instances": [{cat: val}, ...], "diff_cats": [...]}} (개체끼리 값이 다른 속성이 있는 종류만)."""
    p = parse_prompt(prompt)
    multi = {f.split(":", 1)[1] for f in p["flags"] if f.startswith("multi_instance:")}
    if not multi:
        return {}
    M = _mentions(prompt)
    out = {}
    for T in sorted(multi):
        ms = [m for m in M if m["type"] == T]
        insts = [{c: set(v) for c, v in m["attrs"].items()} for m in ms if m["intro"] and m["count"] == 1]
        if not 2 <= len(insts) <= max_instances or any(m["intro"] and m["count"] != 1 for m in ms):
            continue
        ok = True
        for m in ms:
            if m["intro"] or not m["attrs"]:
                continue
            match = [i for i, inst in enumerate(insts)
                     if any(inst.get(c, set()) & v for c, v in m["attrs"].items())
                     and all(not inst.get(c) or inst[c] & v for c, v in m["attrs"].items())]
            if len(match) != 1:
                ok = False
                break
            for c, v in m["attrs"].items():
                insts[match[0]].setdefault(c, set()).update(v)
        if not ok or any(len(v) != 1 for inst in insts for v in inst.values()):
            continue
        insts = [{c: next(iter(v)) for c, v in inst.items()} for inst in insts]
        diff = [c for c in CAT_ORDER if all(c in i for i in insts) and len({i[c] for i in insts}) > 1]
        if diff:
            out[T] = {"instances": insts, "diff_cats": diff}
    return out


def describe(T, inst):
    adj = " ".join(inst[c] for c in CAT_ORDER if c in inst)
    return _np(T, 1, adj or None)


def alternatives(T, g):
    """속성 하나를 개체끼리 돌려 바꾼 설명들 (형제 이미지가 바꾸는 방식)."""
    insts = g["instances"]
    alts = []
    for c in g["diff_cats"]:
        vals = [i[c] for i in insts]
        rot = vals[1:] + vals[:1]
        alts.append([describe(T, {**inst, c: v}) for inst, v in zip(insts, rot)])
    return alts


class InstanceBinder:
    """OwlChecker(여러 상자 검출) + CropColor(SigLIP 잘라 보기)를 써서 이미지마다 개체-속성 결합 여백을 계산한다."""

    def __init__(self, owl_chk, crop, max_boxes=3, box_thr=0.05):
        self.owl, self.crop, self.max_boxes, self.box_thr = owl_chk, crop, max_boxes, box_thr
        self.vec = {}

    def margins(self, items):
        """items: [(image_path, prompt)] → 항목마다 여백(float) 또는 None(해당 없음·상자 부족)."""
        groups = {p: parse_instances(p) for p in dict.fromkeys(p for _, p in items)}
        need = [(path, T) for path, p in items for T in groups[p]]
        self.owl.detect_multi(list(dict.fromkeys(need)), self.max_boxes, self.box_thr)
        phrases = []
        for g in groups.values():
            for T, gg in g.items():
                phrases += [describe(T, i) for i in gg["instances"]] + [x for alt in alternatives(T, gg) for x in alt]
        self.crop._enc_text(phrases)
        todo = {}
        for path, p in items:
            for T, gg in groups[p].items():
                boxes = self.owl.multi_cache.get((path, T), [])[:len(gg["instances"])]
                for b, box in enumerate(boxes):
                    if (path, T, b) not in self.vec:
                        todo[(path, T, b)] = box
        if todo:
            self.vec.update(self.crop._enc_images(list(todo), lambda k: self.crop._crop_image(k[0], todo[k])))
        out = []
        for path, p in items:
            total, used = 0.0, 0
            for T, gg in groups[p].items():
                n = len(gg["instances"])
                V = [self.vec.get((path, T, b)) for b in range(n)]
                if any(v is None for v in V):
                    continue
                V = torch.stack(V)

                def assign(descs):
                    S = V @ torch.stack([self.crop.txt[d] for d in descs]).T        # (상자, 설명)
                    return max(float(sum(S[perm[i], i] for i in range(n))) for perm in itertools.permutations(range(n)))
                total += assign([describe(T, i) for i in gg["instances"]]) - max(assign(a) for a in alternatives(T, gg))
                used += 1
            out.append(total if used else None)
        return out
