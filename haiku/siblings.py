"""형제 그룹(같은 물체 집합) 분석, 비교 쌍 표, 그룹 분할, 학습 배치 샘플러.

torch 없이 동작한다 (numpy, pillow만 사용). 원본 데이터는 읽기만 한다.

실행 (저장소 최상위):
    python -m haiku.siblings --data-dir haikuthon-train

결과:
    reports/siblings/pairs.csv            같은 물체 집합 안의 모든 쌍과 달라진 조건
    reports/siblings/summary.md           정의, 기존 수치 재현, 범주별 집계, 검증 설정 비교, 샘플러 점검
    reports/siblings/summary.json
    reports/siblings/examples.jpg         대표 쌍 이미지 나란히 보기 (육안 확인용)
    reports/siblings/probe_set.csv/.jpg   속성별 점검 세트 (학습 분할 안)
    reports/siblings/contrast_phrases.csv 대조 문구 (판별력 점검 전용, 정답 캡션 아님)
    reports/siblings/manual_checks.csv    이미지를 직접 보고 판정한 결과와 판정자(checked_by) (실행 시 표에 합침)
    splits/split_group_seed42.json        형제 그룹 단위 분할 (그룹 전체가 학습 또는 검증 한쪽)

정의:
    - 물체(type)와 개수(count)를 따로 보관한다. 형제 그룹 키 = 물체 종류 집합(개수 제외).
      그래서 '배낭 1개'와 '배낭 2개'는 같은 그룹이고, 그 차이는 count 조건으로 기록된다.
    - 비교는 텍스트 기준이다. 색은 '어느 물체의 색인지', 관계는 '어느 물체가 어느 물체의 왼쪽·위인지'까지 비교한다.
    - explicit: 두 설명이 같은 대상에 대해 서로 다른 값을 명시.  missing: 한쪽 설명에만 값이 있음.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from haiku.data import load_rows

# ============================================================================ 사전
NOUNS = """person cabinet box bench scarf bowl shelf helmet desk motorcycle suitcase|food container|table block|flower pot
horse stool|roller skate|pencil armchair parrot pitcher chair tomato shirt sofa bed skateboard|picture frame|screwdriver
|slice of pizza|tray pen|kick scooter|rabbit|teddy bear|pumpkin|gardening trowel|bicycle shovel cube cylinder eraser ski ball
boot cookie book|violin bow|ear of corn|sneaker cat rose egg strawberry basketball glove sunflower apple mug dog|paint roller
|umbrella|toy car|handbag saucepan surfboard|bonsai tree|camera wedge|table knife|vase|hand saw|robot toy|tennis racket
|backpack kettle tent|tape measure|colander smartphone cone|acoustic guitar|turtle kite|toy train|baseball bat|spoon
headphones|wall clock|owl hat|guinea pig|bird donut rake paintbrush notebook hamster cushion fern lemon sandwich fork
envelope chicken wrench|soccer ball|broccoli|coat rack|tablet device|watering can|pine tree|cupcake bag|laptop computer
|donkey clipboard sheep|pencil case|stapler pliers snowboard scissors pear|loaf of bread|bus plate duck wristwatch carrot
cactus truck banana dress pyramid goat wheelbarrow dumbbell|drinking glass|palm tree|croissant cow|tennis ball|violin
jacket car ruler sock deer sunglasses fox bottle bear tulip airplane train|frying pan|rug boat hammer|shopping cart
|helicopter penguin cucumber|desk lamp|teapot pig zebra elephant giraffe lamp clock bow"""
ANIMATE = set("""person horse parrot rabbit cat dog turtle owl bird hamster chicken donkey sheep duck goat cow deer fox bear
penguin pig zebra elephant giraffe""".split()) | {"guinea pig"}
IRREGULAR_PLURAL = {"person": "people", "knife": "knives", "shelf": "shelves", "scarf": "scarves", "loaf": "loaves",
                    "cactus": "cacti", "sheep": "sheep", "deer": "deer", "broccoli": "broccoli",
                    "headphones": "headphones", "pliers": "pliers", "scissors": "scissors", "sunglasses": "sunglasses"}
LOCATIONS = """art studio|home office|indoor sports hall|outdoor sports field|sports field|sports hall|living room|dining room
|airport apron|parking lot|train platform|clothing store|snow-covered field|plain backdrop|studio|farm|park|garden
|bedroom|patio|meadow|street|kitchen|office|pasture|forest|classroom|cafe|beach|field|playground|library|orchard
|lakeside|harbor|schoolyard|campsite|garage|workshop|hall|store|background"""
ATTRS = {
    "color": {w: w for w in "beige black blue brown gray green orange pink purple red white yellow".split()}
             | {"grey": "gray", "orange-colored": "orange"},
    "pattern": {w: w for w in "solid-colored striped polka-dotted checkered floral".split()}
               | {"zigzag-patterned": "zigzag", "zigzag": "zigzag", "stripes": "striped"},
    "material": {w: w for w in "ceramic fabric glass leather metal plastic rubber stone wicker cardboard".split()}
                | {"wood": "wooden", "wooden": "wooden"},
    "shape": {w: w for w in "hexagonal oval rectangular round square star-shaped triangular".split()} | {"hexagon": "hexagonal"},
    "state": {w: w for w in "open closed folded unfolded inflated lit unlit".split()},
    "pose": {"lying": "lying", "lies": "lying", "lie": "lying", "sitting": "sitting", "sits": "sitting", "sit": "sitting",
             "standing": "standing", "stands": "standing", "stand": "standing", "kneeling": "kneeling",
             "kneels": "kneeling", "crouching": "crouching", "crouches": "crouching", "sleeping": "sleeping"},
    "size": {"small": "small"},
}
POSE_VERBS = {"lies", "lie", "sits", "sit", "stands", "stand", "kneels", "crouches"}
VERBS = {"is", "are", "rests", "rest", "appears", "appear", "remains", "positioned", "placed", "visible"} | POSE_VERBS
COUNTS = {"one": 1, "two": 2, "three": 3, "single": 1, "a": 1, "an": 1, "another": 1}
DETS = {"the", "this", "its", "their", "other", "both", "each"}
RELS = [  # (토큰열, 관계, 약한 관계 여부: 물체가 바로 오지 않으면 장소·장면 표현으로 보고 버림)
    (("to", "the", "left", "of"), "left", False), (("to", "the", "right", "of"), "right", False),
    (("on", "the", "left", "of"), "left", False), (("on", "the", "right", "of"), "right", False),
    (("left", "of"), "left", False), (("right", "of"), "right", False), (("on", "top", "of"), "on", False),
    (("next", "to"), "next", False), (("in", "front", "of"), "front", False),
    (("above",), "above", False), (("below",), "below", False), (("under",), "below", True),
    (("underneath",), "below", False), (("beneath",), "below", False), (("inside",), "inside", False),
    (("in",), "inside", True), (("on",), "on", True), (("atop",), "on", False), (("beside",), "next", False),
    (("alongside",), "next", False), (("near",), "next", False), (("holds",), "holds", False),
    (("holding",), "holds", False), (("hold",), "holds", False), (("wears",), "wears", False),
    (("wearing",), "wears", False), (("wear",), "wears", False), (("rides",), "rides", False),
    (("riding",), "rides", False), (("ride",), "rides", False), (("contains",), "contains", False),
    (("containing",), "contains", False), (("contain",), "contains", False), (("supports",), "supports", False),
]
SCENE_RELS = {"contains", "holds", "wears", "rides", "supports"}   # 앞에 물체가 없으면 장면 서술("A studio contains")
COMPATIBLE_KINDS = [{"next_to", "left_of"}, {"on", "above"}]        # 동시에 참일 수 있는 관계 종류
OBJECT_CATS = ["count", "color", "pattern", "material", "shape", "state", "pose", "size",
               "relation_direction", "relation_kind"]
SCENE_CATS = ["style", "view", "lighting", "location"]
CAT_KO = {"count": "개수", "color": "색", "pattern": "무늬", "material": "재질", "shape": "모양", "state": "상태",
          "pose": "자세", "size": "크기", "relation_direction": "관계 방향", "relation_kind": "관계 종류",
          "relation_missing": "관계 한쪽만 명시", "style": "화풍", "view": "시점", "lighting": "조명", "location": "장소"}


def _plural(s):
    if " of " in s:
        head, rest = s.split(" of ", 1)
        return _plural(head) + " of " + rest
    *pre, last = s.split()
    if last in IRREGULAR_PLURAL:
        p = IRREGULAR_PLURAL[last]
    elif re.search(r"(s|x|ch|sh)$", last) or last in ("tomato",):
        p = last + "es"
    elif re.search(r"[^aeiou]y$", last):
        p = last[:-1] + "ies"
    else:
        p = last + "s"
    return " ".join(pre + [p])


# NOUNS 문자열에서 "|"로 구분된 조각은 여러 단어 이름, 공백으로 구분된 조각은 한 단어 이름들이다.
def _parse_noun_list():
    out = []
    for chunk in NOUNS.replace("\n", " ").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        words = chunk.split()
        # 조각 안에서 여러 단어 이름은 사전에 있는 것만 인정한다.
        i = 0
        while i < len(words):
            for L in (4, 3, 2):
                cand = " ".join(words[i:i + L])
                if cand in MULTIWORD:
                    out.append(cand)
                    i += L
                    break
            else:
                out.append(words[i])
                i += 1
    return out


MULTIWORD = {"food container", "flower pot", "roller skate", "picture frame", "slice of pizza", "kick scooter",
             "teddy bear", "gardening trowel", "violin bow", "ear of corn", "paint roller", "toy car", "bonsai tree",
             "table knife", "hand saw", "robot toy", "tennis racket", "tape measure", "acoustic guitar", "toy train",
             "baseball bat", "wall clock", "guinea pig", "soccer ball", "coat rack", "tablet device", "watering can",
             "pine tree", "laptop computer", "pencil case", "loaf of bread", "drinking glass", "palm tree",
             "tennis ball", "frying pan", "shopping cart", "desk lamp"}


def build_phrase_table():
    table = {}
    for n in _parse_noun_list():
        table[tuple(n.split())] = ("NOUN", n, False)
        table[tuple(_plural(n).split())] = ("NOUN", n, True)
    table[("oranges",)] = ("NOUN", "orange fruit", True)
    for loc in LOCATIONS.replace("\n", "").split("|"):
        loc = loc.strip()
        if loc:
            table[tuple(loc.split())] = ("LOC", loc, False)
    return table


PHRASES = build_phrase_table()
MAX_PHRASE = max(len(k) for k in PHRASES)
ATTR_LOOKUP = {w: (cat, v) for cat, m in ATTRS.items() for w, v in m.items()}
REL_LOOKUP = sorted(RELS, key=lambda r: -len(r[0]))


# ============================================================================ 파싱
def _tokenize(prompt):
    t = prompt.lower().replace("’", "'").replace("\n", " ")
    t = re.sub(r"\bpairs? of\s+", "", t)            # "one pair of headphones" → "one headphones"
    t = re.sub(r"\b(lit|illuminated) (by|with|from)\b", r"LITBY \2", t)   # 조명 표현의 lit은 상태가 아님
    t = re.sub(r"\bside-lit\b", "sidelit", t)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|'s|[,.;:]|LITBY", t)


def _typed_tokens(words):
    out, i = [], 0
    while i < len(words):
        w = words[i]
        hit = None
        for L in range(min(MAX_PHRASE, len(words) - i), 0, -1):
            key = tuple(words[i:i + L])
            if key in PHRASES:
                hit = (PHRASES[key], L)
                break
        if hit:
            (kind, name, plural), L = hit
            out.append({"t": kind, "type": name, "plural": plural, "i": len(out)})
            i += L
            continue
        rel = next((r for r in REL_LOOKUP if tuple(words[i:i + len(r[0])]) == r[0]), None)
        if rel:
            out.append({"t": "REL", "rel": rel[1], "weak": rel[2], "w": " ".join(rel[0])})
            i += len(rel[0])
            continue
        if w in (",",):
            out.append({"t": "PUNCT", "w": w})
        elif w == "'s":
            out.append({"t": "POSS", "w": w})
        elif w in COUNTS:
            out.append({"t": "COUNT", "n": COUNTS[w], "w": w})
        elif w in ATTR_LOOKUP:
            cat, val = ATTR_LOOKUP[w]
            out.append({"t": "ATTR", "cat": cat, "val": val, "w": w, "verb": w in POSE_VERBS})
        else:
            out.append({"t": "W", "w": w})
        i += 1
    for k, tok in enumerate(out):
        tok["i"] = k
    return out


def _sentences(words):
    cur = []
    for w in words:
        if w in (".", ";", ":"):
            if cur:
                yield cur
            cur = []
        else:
            cur.append(w)
    if cur:
        yield cur


def _is_verbish(tok):
    return (tok["t"] == "W" and tok["w"] in VERBS) or (tok["t"] == "ATTR" and tok.get("verb"))


def _is_subject(S, i):
    """명사 뒤에 (속성·쉼표를 건너뛰고) 서술어가 오면 절의 주어. "X and is ..."의 X는 주어가 아니다(앞 절의 목적어)."""
    k = i + 1
    while k < len(S):
        tok = S[k]
        if tok["t"] == "PUNCT" or (tok["t"] == "ATTR" and not tok.get("verb")):
            k += 1
            continue
        if tok["t"] == "W" and tok["w"] == "and":
            nxt = S[k + 1] if k + 1 < len(S) else None
            if nxt is not None and nxt["t"] == "ATTR" and not nxt.get("verb"):   # "black and sitting"
                k += 1
                continue
            return False
        return _is_verbish(tok)
    return False


def _scene_fields(text):
    t = text.lower().replace("\n", " ")
    style = None
    for pat, name in [(r"clay-style", "clay 3D"), (r"low-poly", "low-poly 3D"), (r"paper-cut", "paper-cut collage"),
                      (r"colored-pencil", "colored-pencil"), (r"pencil sketch|sketch", "pencil sketch"),
                      (r"oil painting", "oil painting"), (r"watercolor", "watercolor"), (r"gouache", "gouache"),
                      (r"flat illustration", "flat illustration"), (r"\bink\b", "ink"),
                      (r"photograph", "photograph"), (r"\b3d\b|render", "3D render")]:
        if re.search(pat, t):
            style = name
            break
    lighting = None
    for pat, name in [(r"side lighting|side-lit|lit from the side", "side"), (r"warm indoor", "warm indoor"),
                      (r"soft diffuse|diffuse lighting", "soft diffuse"), (r"direct sunlight", "direct sunlight")]:
        if re.search(pat, t):
            lighting = name
            break
    t2 = re.sub(r"side lighting|side-lit|lit from the side", " ", t)
    view = None
    for pat, name in [(r"overhead", "overhead"), (r"oblique", "oblique"), (r"\bfront\b|front-view", "front"),
                      (r"\bside\b|side-view", "side")]:
        if re.search(pat, t2):
            view = name
            break
    return {"style": style, "view": view, "lighting": lighting}


def parse_prompt(prompt):
    """설명 하나를 물체(종류·개수·속성)와 관계로 바꾼다. 규칙 기반 근사치이며, 애매한 곳은 flags에 남긴다."""
    words = _tokenize(prompt)
    objects = defaultdict(lambda: {"count": 0, "intros": 0, "plural_seen": False,
                                   "attrs": defaultdict(set)})
    relations, flags, location = set(), [], None
    for sent in _sentences(words):
        S = _typed_tokens(sent)
        noun_idx = [k for k, tok in enumerate(S) if tok["t"] == "NOUN"]
        for k, tok in enumerate(S):
            if tok["t"] == "LOC" and location is None and tok["type"] not in ("plain backdrop", "background"):
                location = tok["type"]
        # --- 물체 소개와 개수
        for k in noun_idx:
            tok = S[k]
            o = objects[tok["type"]]
            o["plural_seen"] |= tok["plural"]
            j = k - 1
            while j >= 0 and (S[j]["t"] in ("ATTR", "PUNCT") or (S[j]["t"] == "W" and S[j]["w"] in ("other", "and"))):
                j -= 1
            if j >= 0 and S[j]["t"] == "COUNT" and not (j > 0 and S[j - 1]["t"] == "W" and S[j - 1]["w"] == "the"):
                n = S[j]["n"]
                if S[j]["w"] == "single":                 # "two single gloves" → 2 ("single"은 낱개라는 뜻)
                    j2 = j - 1
                    while j2 >= 0 and S[j2]["t"] in ("ATTR", "PUNCT"):
                        j2 -= 1
                    if j2 >= 0 and S[j2]["t"] == "COUNT" and S[j2]["w"] != "single":
                        n = S[j2]["n"]
                o["count"] += n
                o["intros"] += 1
        # 소개 문구 뒤 이름 없는 단어(사전에 없는 물체 후보)
        for k, tok in enumerate(S):
            if tok["t"] == "COUNT" and tok["w"] in ("one", "two", "three", "single"):
                j = k + 1
                while j < len(S) and S[j]["t"] in ("ATTR", "PUNCT"):
                    j += 1
                if j < len(S) and S[j]["t"] == "W" and S[j]["w"] not in ("other", "of", "and", "the"):
                    flags.append(f"unknown_noun:{S[j]['w']}")
        # --- 속성 연결
        for q, tok in enumerate(S):
            if tok["t"] != "ATTR":
                continue
            target = None
            j = q + 1
            while j < len(S) and (S[j]["t"] in ("PUNCT",) or (S[j]["t"] == "ATTR" and not S[j].get("verb"))
                                  or (S[j]["t"] == "W" and S[j]["w"] == "and")):
                j += 1
            if j < len(S) and S[j]["t"] == "NOUN" and not tok.get("verb"):
                target = S[j]["type"]                                   # 명사 앞 수식
            else:
                before = [k for k in noun_idx if k < q]
                if before:
                    nb = before[-1]
                    between = [x.get("w") for x in S[nb + 1:q]]
                    if "and" in between and ("is" in between or "are" in between):
                        subj = [k for k in noun_idx if k < q and _is_subject(S, k)]
                        nb = subj[-1] if subj else noun_idx[0]
                    target = S[nb]["type"]                              # 명사 뒤 서술·동격
            if target is None:
                if tok["cat"] not in ("pose",):
                    flags.append(f"unbound_attr:{tok['w']}")
                continue
            if tok["cat"] == "pose" and target not in ANIMATE:
                continue                                                 # 무생물의 sits/rests는 배치 동사
            objects[target]["attrs"][tok["cat"]].add(tok["val"])
        # --- 소유격 관계: "to the parrot's left"
        for k, tok in enumerate(S):
            if tok["t"] == "POSS" and k >= 1 and S[k - 1]["t"] == "NOUN" and k + 1 < len(S) and S[k + 1]["t"] == "W" \
                    and S[k + 1]["w"] in ("left", "right"):
                obj = S[k - 1]["type"]
                before = [x for x in noun_idx if x < k - 1]
                if before:
                    relations.add(_canon(S[k + 1]["w"], S[before[-1]]["type"], obj))
                else:
                    flags.append("rel_no_subject:poss")
        # --- 관계
        for p, tok in enumerate(S):
            if tok["t"] != "REL":
                continue
            if p + 1 < len(S) and S[p + 1]["t"] == "W" and S[p + 1]["w"] in ("which", "whom"):
                # 도치 관계절 "one cabinet, next to which is one pear" → pear next_to cabinet
                before = [k for k in noun_idx if k < p]
                after = [k for k in noun_idx if k > p]
                if before and after and S[before[-1]]["type"] != S[after[0]]["type"]:
                    relations.add(_canon(tok["rel"], S[after[0]]["type"], S[before[-1]]["type"]))
                else:
                    flags.append(f"rel_no_subject:{tok['w']}")
                continue
            j = p + 1
            while j < len(S) and ((S[j]["t"] == "W" and S[j]["w"] in DETS) or S[j]["t"] in ("COUNT",)
                                  or (S[j]["t"] == "ATTR" and not S[j].get("verb")) or S[j]["t"] == "PUNCT"):
                j += 1
            if not (j < len(S) and S[j]["t"] == "NOUN"):
                if not tok["weak"] and tok["rel"] not in ("holds", "wears", "rides") and not (
                        j < len(S) and S[j]["t"] == "LOC"):
                    flags.append(f"rel_no_object:{tok['w']}")
                continue
            objs = [j]
            # 등위 목적어 "above one frame and one sunflower" (뒤에 다른 서술어가 없을 때만)
            m = j + 1
            while m < len(S):
                while m < len(S) and (S[m]["t"] in ("ATTR", "PUNCT") and not S[m].get("verb")):
                    m += 1
                if m < len(S) and S[m]["t"] == "W" and S[m]["w"] == "and":
                    n_ = m + 1
                    while n_ < len(S) and ((S[n_]["t"] == "W" and S[n_]["w"] in DETS) or S[n_]["t"] in ("COUNT", "ATTR")):
                        n_ += 1
                    if n_ < len(S) and S[n_]["t"] == "NOUN" and not any(_is_verbish(x) or x["t"] == "REL" for x in S[n_ + 1:]):
                        objs.append(n_)
                        m = n_ + 1
                        continue
                break
            before = [k for k in noun_idx if k < p]
            subj = None
            if before:
                rel_pron = [k for k in range(before[-1] + 1, p) if S[k]["t"] == "W" and S[k]["w"] in ("which", "that", "who")]
                if rel_pron:
                    subj = before[-1]
                else:
                    subs = [k for k in before if _is_subject(S, k)]
                    subj = subs[-1] if subs else before[0]
            elif tok["rel"] in SCENE_RELS:
                continue
            else:                                                        # 도치: "To the right of one X is one Y"
                after = [k for k in noun_idx if k > objs[-1]]
                if after:
                    subj = after[0]
                else:
                    flags.append(f"rel_no_subject:{tok['w']}")
                    continue
            for o in objs:
                a, b = S[subj]["type"], S[o]["type"]
                if a == b:
                    flags.append(f"self_relation:{a}")
                    continue
                relations.add(_canon(tok["rel"], a, b))
    # --- 마무리
    out_objects = {}
    for name, o in objects.items():
        count = o["count"]
        if o["intros"] == 0:
            count = 2 if o["plural_seen"] else 1
            flags.append(f"count_inferred:{name}")
        if o["intros"] > 1:
            flags.append(f"multi_instance:{name}")
        out_objects[name] = {"count": count, "attrs": {c: sorted(v) for c, v in o["attrs"].items()}}
    scene = _scene_fields(prompt)
    scene["location"] = location
    return {"objects": out_objects, "relations": sorted(relations), "scene": scene, "flags": sorted(set(flags))}


def _canon(rel, a, b):
    if rel == "left":
        return ("left_of", a, b)
    if rel == "right":
        return ("left_of", b, a)
    if rel == "above":
        return ("above", a, b)
    if rel == "below":
        return ("above", b, a)
    if rel == "on":
        return ("on", a, b)
    if rel == "supports":
        return ("on", b, a)
    if rel == "inside":
        return ("inside", a, b)
    if rel == "contains":
        return ("inside", b, a)
    if rel == "next":
        return ("next_to",) + tuple(sorted((a, b)))
    if rel == "front":
        return ("front_of", a, b)
    return (rel, a, b)                                                    # holds, wears, rides


def group_key(parsed):
    return frozenset(parsed["objects"]) if parsed["objects"] else None


# ============================================================================ 비교
def compare(pa, pb):
    """같은 물체 집합의 두 설명을 비교한다. 반환: 차이 목록 (category, level, target, a, b)."""
    diffs = []
    for name in sorted(set(pa["objects"]) | set(pb["objects"])):
        oa, ob = pa["objects"].get(name), pb["objects"].get(name)
        if not oa or not ob:
            continue
        if oa["count"] != ob["count"]:
            diffs.append(("count", "explicit", name, oa["count"], ob["count"]))
        for cat in ["color", "pattern", "material", "shape", "state", "pose", "size"]:
            va, vb = set(oa["attrs"].get(cat, [])), set(ob["attrs"].get(cat, []))
            if va == vb:
                continue
            level = "explicit" if va and vb else "missing"
            diffs.append((cat, level, name, "/".join(sorted(va)) or "-", "/".join(sorted(vb)) or "-"))
    ra, rb = defaultdict(set), defaultdict(set)
    for r in pa["relations"]:
        ra[frozenset(r[1:])].add(r)
    for r in pb["relations"]:
        rb[frozenset(r[1:])].add(r)
    for pair in sorted(set(ra) | set(rb), key=lambda s: sorted(s)):
        A, B = ra.get(pair, set()), rb.get(pair, set())
        if A == B:
            continue
        label = "–".join(sorted(pair))
        fa, fb = _fmt_rel(A), _fmt_rel(B)
        if A and B:
            reversed_ = any((r[0], r[2], r[1]) in B for r in A if r[0] != "next_to")
            kinds_a, kinds_b = {r[0] for r in A}, {r[0] for r in B}
            if reversed_:
                diffs.append(("relation_direction", "explicit", label, fa, fb))
            else:
                compatible = any(kinds_a | kinds_b <= c for c in COMPATIBLE_KINDS)
                diffs.append(("relation_kind", "compatible" if compatible else "explicit", label, fa, fb))
        else:
            diffs.append(("relation_missing", "missing", label, fa or "-", fb or "-"))
    for cat in SCENE_CATS:
        va, vb = pa["scene"].get(cat), pb["scene"].get(cat)
        if va != vb:
            diffs.append((cat, "explicit" if va and vb else "missing", "scene", va or "-", vb or "-"))
    return diffs


def _fmt_rel(rs):
    return "; ".join(f"{a} {r} {b}" for r, a, b in sorted(rs))


def classify_pair(pa, pb, diffs):
    explicit = [d for d in diffs if d[1] == "explicit"]
    explicit_obj = [d for d in explicit if d[0] in OBJECT_CATS]
    reasons = []
    flags = set(pa["flags"]) | set(pb["flags"])
    hard_flags = sorted(f for f in flags if f.split(":")[0] in
                        ("multi_instance", "rel_no_subject", "rel_no_object", "self_relation", "unbound_attr", "unknown_noun"))
    if hard_flags:
        reasons.append("파싱 애매: " + ", ".join(hard_flags[:4]))
    if not explicit:
        reasons.append("명시적 차이 없음: 같은 장면을 다르게 쓴 문장이거나 파싱이 놓친 차이")
    elif not explicit_obj:
        reasons.append("물체 조건은 같고 장면(화풍·시점·조명·장소)만 다름")
    if any(d[1] == "compatible" for d in diffs):
        reasons.append("관계 종류가 달라도 동시에 참일 수 있음(next_to/left_of, on/above)")
    if any(d[1] == "missing" for d in diffs) and not explicit_obj:
        reasons.append("한쪽 설명에만 있는 조건이 있음(누락)")
    n = len(explicit)
    # 한 가지 속성만 명확히 다른가: 명시적 차이가 물체 조건 1개뿐이고, 한쪽에만 적힌 조건·동시 참 가능 관계·파싱 애매가 없어야 '확정'
    unverifiable = any(d[1] in ("missing", "compatible") for d in diffs) or bool(hard_flags)
    if n == 1 and len(explicit_obj) == 1:
        single = "단일(미확인)" if unverifiable else "단일 확정"
    else:
        single = "해당 없음"
    return {
        "n_explicit": n, "n_explicit_object": len(explicit_obj),
        "diff_class": "단일" if n == 1 else ("복수" if n > 1 else "명시 차이 없음"),
        "single_status": single,
        "clear": bool(explicit_obj) and not hard_flags,
        "object_categories": sorted({d[0] for d in explicit_obj}),
        "binding_swap": binding_swap(pa, pb),
        "ambiguous_reason": " / ".join(reasons),
    }


def _vals(p, name, cat):
    if cat == "count":
        return (str(p["objects"][name]["count"]),)
    return tuple(p["objects"][name]["attrs"].get(cat, []))


def swap_pairs(pa, pb, cat):
    """cat 값이 두 물체 사이에서 서로 바뀐 물체 쌍 목록 [(n1, n2)]."""
    names = sorted(set(pa["objects"]) & set(pb["objects"]))
    out = []
    for x, n1 in enumerate(names):
        for n2 in names[x + 1:]:
            a1, a2, b1, b2 = _vals(pa, n1, cat), _vals(pa, n2, cat), _vals(pb, n1, cat), _vals(pb, n2, cat)
            if a1 and a2 and a1 != a2 and a1 == b2 and a2 == b1:
                out.append((n1, n2))
    return out


def binding_swap(pa, pb):
    """두 물체의 색 또는 개수가 서로 바뀐 경우 (예: 파란 배낭·보라 벤치 ↔ 보라 배낭·파란 벤치). 실제 두 설명에서만 찾는다."""
    out = []
    for cat, ko in (("color", "색"), ("count", "개수")):
        for n1, n2 in swap_pairs(pa, pb, cat):
            f = lambda p, n: "/".join(_vals(p, n, cat))
            out.append(f"{ko}: {n1}={f(pa, n1)}·{n2}={f(pa, n2)} ↔ {n1}={f(pb, n1)}·{n2}={f(pb, n2)}")
    return " | ".join(out)


# ============================================================================ 대조 문구 (판별력 점검 전용, 정답 캡션 아님)
NUM_WORD = {1: "one", 2: "two", 3: "three"}


def _np(name, count=1, adj=None):
    """명사구: count=1이면 관사, 2 이상이면 수사 + 복수."""
    noun = name if count == 1 else _plural(name)
    head = f"{adj} {noun}" if adj else noun
    if count == 1:
        return ("an " if head[0] in "aeiou" else "a ") + head
    return f"{NUM_WORD.get(count, str(count))} {head}"


def contrast_phrases(pa, pb, diffs, id_a, id_b):
    """명시적 차이 1개짜리 쌍에서, 두 실제 설명의 값으로 목표·대조 문구를 만든다.
    반환 행마다 target_image(그 문구에 맞는 이미지)를 적는다. 의미를 규칙으로 확정하기 어려우면 빈 목록."""
    out = []
    explicit = [d for d in diffs if d[1] == "explicit"]
    multi = {f.split(":", 1)[1] for f in pa["flags"] + pb["flags"] if f.startswith("multi_instance:")}
    # 색 결합 교차: 색이 바뀐 물체가 정확히 두 개이고 서로 맞바뀐 경우만
    changed = [n for n in sorted(set(pa["objects"]) & set(pb["objects"])) if _vals(pa, n, "color") != _vals(pb, n, "color")]
    sw = swap_pairs(pa, pb, "color")
    if len(changed) == 2 and sw == [tuple(changed)] and not (set(changed) & multi) \
            and all(len(_vals(p, n, "color")) == 1 for p in (pa, pb) for n in changed) \
            and all(pa["objects"][n]["count"] == pb["objects"][n]["count"] for n in changed):
        n1, n2 = changed
        ph = lambda p: " and ".join(_np(n, p["objects"][n]["count"], _vals(p, n, "color")[0]) for n in (n1, n2))
        out.append({"category": "color_binding_swap", "target_image": id_a, "target": ph(pa), "contrast": ph(pb)})
    # 개수 결합 교차: 개수가 바뀐 물체가 정확히 두 개이고 서로 맞바뀐 경우만
    changed = [n for n in sorted(set(pa["objects"]) & set(pb["objects"])) if pa["objects"][n]["count"] != pb["objects"][n]["count"]]
    if len(changed) == 2 and swap_pairs(pa, pb, "count") == [tuple(changed)] and not (set(changed) & multi):
        n1, n2 = changed
        ph = lambda p: " and ".join(_np(n, p["objects"][n]["count"]) for n in (n1, n2))
        out.append({"category": "count_binding_swap", "target_image": id_a, "target": ph(pa), "contrast": ph(pb)})
    if len(explicit) != 1:
        return out
    cat, _, target, va, vb = explicit[0]
    if cat in ("color", "pattern", "material", "shape", "state", "pose", "size"):
        if "/" in va or "/" in vb:                    # 한 물체에 값이 여러 개면 문구로 확정하지 않음
            return out
        if any(f.startswith(f"multi_instance:{target}") for f in pa["flags"] + pb["flags"]):
            return out
        ca, cb = pa["objects"][target]["count"], pb["objects"][target]["count"]
        if ca != cb:
            return out
        out.append({"category": cat, "target_image": id_a, "target": _np(target, ca, va), "contrast": _np(target, ca, vb)})
    elif cat == "count":
        out.append({"category": cat, "target_image": id_a, "target": _np(target, int(va)), "contrast": _np(target, int(vb))})
    elif cat == "relation_direction":
        ra = [r for r in pa["relations"] if frozenset(r[1:]) == frozenset(target.split("–"))]
        if len(ra) != 1 or ra[0][0] not in ("left_of", "above"):
            return out
        rel, x, y = ra[0]
        cx, cy = pa["objects"][x]["count"], pa["objects"][y]["count"]
        word, opp = ("to the left of", "to the right of") if rel == "left_of" else ("above", "below")
        verb_np = _np(x, cx)
        out.append({"category": "relation_left_right" if rel == "left_of" else "relation_above_below",
                    "target_image": id_a, "target": f"{verb_np} {word} {_np(y, cy)}",
                    "contrast": f"{verb_np} {opp} {_np(y, cy)}"})
    for row in out:
        row["use"] = "probe_only (정답 캡션 아님)"
        row["other_image"] = id_b
    return out


# ============================================================================ 그룹·쌍
def build_groups(rows, parsed):
    by_key = defaultdict(list)
    for r in rows:
        k = group_key(parsed[r["id"]])
        by_key[k if k is not None else ("__unparsed__", r["id"])].append(r["id"])
    ordered = sorted(by_key.values(), key=lambda v: int(min(v, key=int)))
    group_of, members = {}, {}
    for n, ids in enumerate(ordered, 1):
        gid = f"G{n:04d}"
        members[gid] = sorted(ids, key=int)
        for i in ids:
            group_of[i] = gid
    return group_of, members


def build_pairs(rows, parsed, members):
    prompts = {r["id"]: r["prompt"] for r in rows}
    out = []
    for gid, ids in members.items():
        for x, a in enumerate(ids):
            for b in ids[x + 1:]:
                d = compare(parsed[a], parsed[b])
                c = classify_pair(parsed[a], parsed[b], d)
                out.append({"group": gid, "a": a, "b": b, "prompt_a": prompts[a], "prompt_b": prompts[b],
                            "diffs": d, **c})
    return out


def fmt_diffs(diffs, level=None):
    sel = [d for d in diffs if level is None or d[1] == level]
    return "; ".join(f"{CAT_KO.get(c, c)}[{t}] {a}→{b}" for c, lv, t, a, b in sel)


# ============================================================================ 기존 수치 재현 (단어 집합 방식)
LEGACY_CAT = {
    "색": "brown white black yellow green red beige orange gray grey blue purple pink".split(),
    "무늬": "solid-colored striped polka-dotted checkered floral patterned plaid".split(),
    "재질": "leather plastic metal metallic wooden wood glass fabric ceramic paper stone rubber wool cardboard".split(),
    "자세·상태": "lying sitting standing kneeling crouching running walking jumping open closed".split(),
    "관계": "left right above below next inside top under beneath near holds holding wears wearing rides riding resting rests on behind".split(),
    "개수": "one two three single".split(),
    "모양": "square round oval rectangular circular triangular cylindrical".split(),
    "화풍": "photograph photographed render rendered 3d illustration watercolor painting oil colored-pencil drawing gouache low-poly paper-cut collage clay-style pencil sketch ink flat".split(),
    "시점": "overhead front side oblique".split(),
    "조명": "warm indoor soft diffuse direct sunlight lighting lit".split(),
}
LEGACY_LOC = set("farm park living garden bedroom patio dining meadow street kitchen office airport apron pasture parking lot forest classroom train platform cafe beach field snow-covered desert library bathroom playground lake harbor road sidewalk courtyard warehouse workshop greenhouse rooftop".split())


def legacy_counts(rows):
    """2026-11-21 대화에서 보고한 수치(관계 1,137 / 색 713)를 같은 방식으로 다시 계산한다."""
    w2c = {w: c for c, ws in LEGACY_CAT.items() for w in ws}
    stop = set("a an the is are and which that it its with in of to from as by at for this there who shows shown show "
               "appears appear scene image setting set sits stands positioned present also view viewed under has have "
               "pattern made uses use be both while each other one's".split()) - set(w2c)
    stop |= {"view", "viewed", "lighting", "light", "contains", "containing", "flower", "plain", "backdrop", "studio",
             "angle", "pair", "room"}

    def content(t):
        return set(w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower()) if w not in stop)

    P = {r["id"]: r["prompt"] for r in rows}
    G = defaultdict(list)
    for i in sorted(P, key=int):
        G[frozenset({w for w in content(P[i]) if w not in w2c} - LEGACY_LOC)].append(i)
    pairs = [(a, b) for v in G.values() if len(v) > 1 for k, a in enumerate(v) for b in v[k + 1:]]
    single, overlap, none = Counter(), Counter(), 0
    for a, b in pairs:
        ca, cb = content(P[a]), content(P[b])
        cats = {w2c.get(w, "객체·장소") for w in ca ^ cb}
        if not cats:
            none += 1
            continue
        overlap.update(cats)
        if len(cats) == 1:
            single[next(iter(cats))] += 1
    return {"scope": "공개 train 4,000쌍 전체 (분할 구분 없음)", "n_groups": len(G),
            "n_with_siblings": sum(len(v) for v in G.values() if len(v) > 1), "n_pairs": len(pairs),
            "pair_counting": "같은 그룹 안에서 A<B 순서로 한 번만 셈 (A–B와 B–A 중복 없음)",
            "single_category_exclusive": dict(single.most_common()),
            "category_overlapping": dict(overlap.most_common()), "no_content_difference": none}


# ============================================================================ 그룹 분할
def group_split(rows, members, seed=42, valid_ratio=0.2):
    ids = [r["id"] for r in rows]
    target = round(len(ids) * valid_ratio)
    gids = sorted(members)
    random.Random(seed).shuffle(gids)
    valid = []
    for g in gids:
        if len(valid) + len(members[g]) <= target:
            valid.extend(members[g])
    valid = sorted(valid, key=int)
    vs = set(valid)
    train = [i for i in sorted(ids, key=int) if i not in vs]
    return train, valid


def check_split(train, valid, group_of):
    t, v = set(train), set(valid)
    crossing = sorted({group_of[i] for i in v} & {group_of[i] for i in t})
    return {"overlap_ids": len(t & v), "groups_crossing": len(crossing), "crossing_examples": crossing[:5]}


# ============================================================================ 검증 후보 구성 통계
def sibling_stats(query_ids, gallery_ids, group_of):
    gal = set(gallery_ids)
    by_group = defaultdict(set)
    for g in gallery_ids:
        by_group[group_of[g]].add(g)
    counts = [len(by_group[group_of[q]] - {q}) for q in query_ids]
    assert all(q in gal for q in query_ids), "정답 이미지가 후보에 없는 질의가 있습니다."
    n = len(counts)
    return {"n_queries": n, "n_gallery": len(gal),
            "mean_siblings_per_query": round(sum(counts) / n, 3),
            "queries_with_sibling": sum(c > 0 for c in counts),
            "ratio_with_sibling": round(sum(c > 0 for c in counts) / n, 4),
            "max_siblings": max(counts), "per_query": dict(zip(query_ids, counts))}


# ============================================================================ 샘플러 (A/B 공통 규칙)
def batch_sizes(n, bs):
    return [bs] * (n // bs) + ([n % bs] if n % bs else [])


def random_batches(ids, bs, rng):
    """A: 무작위 배치. 한 에포크에 모든 ID가 정확히 한 번."""
    order = list(ids)
    rng.shuffle(order)
    out, s = [], 0
    for size in batch_sizes(len(order), bs):
        out.append(order[s:s + size])
        s += size
    return out


def sibling_batches(ids, clear_pairs, bs, pairs_per_batch, rng):
    """B: 배치마다 형제 비교 쌍을 최대 pairs_per_batch개 넣고 나머지는 일반 샘플로 채운다.
    한 에포크에 모든 ID가 정확히 한 번(A와 같은 ID 집합·노출 횟수·배치 크기)."""
    idset = set(ids)
    cand = [(a, b) for a, b in clear_pairs if a in idset and b in idset]
    rng.shuffle(cand)
    sizes = batch_sizes(len(ids), bs)
    capacity = sum(min(pairs_per_batch, s // 2) for s in sizes)
    used, chosen = set(), []
    for a, b in cand:
        if len(chosen) >= capacity:
            break
        if a not in used and b not in used:
            chosen.append((a, b))
            used.update((a, b))
    rest = [i for i in ids if i not in used]
    rng.shuffle(rest)
    batches, pi, ri = [], 0, 0
    for size in sizes:
        k = min(pairs_per_batch, size // 2, len(chosen) - pi)
        b = [x for pair in chosen[pi:pi + k] for x in pair]
        pi += k
        b += rest[ri:ri + size - len(b)]
        ri += size - 2 * k
        rng.shuffle(b)
        batches.append(b)
    rng.shuffle(batches)
    assert sorted(x for b in batches for x in b) == sorted(ids)
    return batches, chosen


def batch_pair_stats(batches, group_of, pair_index):
    """실제 배치 안에 같이 들어간 형제 쌍 수와 종류."""
    n_sib, n_clear, cats = 0, 0, Counter()
    for b in batches:
        for x, a in enumerate(b):
            for c in b[x + 1:]:
                if group_of[a] != group_of[c]:
                    continue
                n_sib += 1
                info = pair_index.get((a, c)) or pair_index.get((c, a))
                if info and info["clear"]:
                    n_clear += 1
                    cats.update(info["object_categories"])
    return {"n_batches": len(batches), "sibling_pairs_in_batches": n_sib, "clear_pairs_in_batches": n_clear,
            "clear_pairs_per_batch": round(n_clear / len(batches), 3), "categories": dict(cats.most_common())}


# ============================================================================ 대표 쌍 그림
def save_examples(pairs, images_dir, path, per_cat=3, seed=0, thumb=256):
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    picks = []
    for cat in ["color", "count", "relation_direction", "relation_kind", "pattern", "material", "pose", "state", "shape"]:
        pool = [p for p in pairs if p["clear"] and p["n_explicit"] == 1 and p["object_categories"] == [cat]]
        picks += [(cat, p) for p in rng.sample(pool, min(per_cat, len(pool)))]
    pool = [p for p in pairs if p["diff_class"] == "명시 차이 없음"]
    picks += [("none", p) for p in rng.sample(pool, min(per_cat, len(pool)))]
    pad, text_h = 8, 120
    W = 2 * thumb + 3 * pad
    H = len(picks) * (thumb + text_h + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for n, (cat, p) in enumerate(picks, 1):
        for k, key in enumerate(("a", "b")):
            with Image.open(Path(images_dir) / f"{p[key]}.png") as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (pad + k * (thumb + pad), y))
        lines = [f"#{n} [{cat}] {p['a']} vs {p['b']} ({p['group']})", fmt_diffs(p["diffs"], "explicit")[:110]]
        for key in ("a", "b"):
            t = p[f"prompt_{key}"].replace("\n", " ")
            lines += [f"{p[key]}: " + t[:80], "      " + t[80:160]]
        for k, line in enumerate(lines):
            draw.text((pad, y + thumb + 4 + k * 15), line, fill="black")
        y += thumb + text_h + pad
    sheet.save(path, quality=85)
    return [{"n": n, "category": cat, "a": p["a"], "b": p["b"], "group": p["group"],
             "explicit": fmt_diffs(p["diffs"], "explicit")} for n, (cat, p) in enumerate(picks, 1)]


# ============================================================================ 실행
def run(data_dir, report_dir="reports/siblings", split_dir="splits", official_split="splits/split_seed42.json",
        seed=42, bs=16, pairs_per_batch=4):
    data_dir, report_dir, split_dir = Path(data_dir), Path(report_dir), Path(split_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(data_dir)
    parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
    group_of, members = build_groups(rows, parsed)
    pairs = build_pairs(rows, parsed, members)
    pair_index = {(p["a"], p["b"]): p for p in pairs}

    # 이미지를 직접 보고 판정한 결과 (manual_checks.csv, checked_by 열에 판정자) — 다시 실행해도 유지된다
    checks = {}
    mc = report_dir / "manual_checks.csv"
    if mc.exists():
        for r in csv.DictReader(mc.open(encoding="utf-8-sig")):
            checks[(r["id_a"], r["id_b"])] = (r["image_check"], r.get("note", ""), r.get("checked_by", ""))

    # 1) 쌍 표
    with (report_dir / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "id_a", "id_b", "prompt_a", "prompt_b", "diff_class", "single_status", "n_explicit",
                    "n_explicit_object", "clear_for_sampler", "object_categories", "binding_swap", "explicit_diffs",
                    "missing_diffs", "compatible_diffs", "ambiguous_reason", "image_check", "image_check_note", "checked_by"])
        for p in pairs:
            chk = checks.get((p["a"], p["b"]), ("", "", ""))
            w.writerow([p["group"], p["a"], p["b"], p["prompt_a"].replace("\n", " "), p["prompt_b"].replace("\n", " "),
                        p["diff_class"], p["single_status"], p["n_explicit"], p["n_explicit_object"], p["clear"],
                        "|".join(p["object_categories"]), p["binding_swap"], fmt_diffs(p["diffs"], "explicit"),
                        fmt_diffs(p["diffs"], "missing"), fmt_diffs(p["diffs"], "compatible"), p["ambiguous_reason"],
                        chk[0], chk[1], chk[2]])
    with (report_dir / "parsed.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "group": group_of[r["id"]], **parsed[r["id"]]}, ensure_ascii=False) + "\n")

    # 2) 분할
    train, valid = group_split(rows, members, seed)
    leak = check_split(train, valid, group_of)
    split = {"name": f"split_group_seed{seed}", "created_at": datetime.now().isoformat(timespec="seconds"),
             "seed": seed, "method": "형제 그룹(물체 종류 집합, 개수 제외) 단위로 무작위 배정. 검증 목표 20%",
             "n_train": len(train), "n_valid": len(valid),
             "n_groups_train": len({group_of[i] for i in train}), "n_groups_valid": len({group_of[i] for i in valid}),
             "leak_check": leak,
             "rules": ["검증 ID는 학습·증강·hard negative 채굴에 쓰지 않는다."],
             "train_ids": train, "valid_ids": valid, "group_of": group_of}
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"split_group_seed{seed}.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")

    # 3) 검증 설정별 형제 오답 수
    off = json.loads(Path(official_split).read_text(encoding="utf-8"))
    off_train, off_valid = off["train_ids"], off["valid_ids"]
    extra = [r["id"] for r in random.Random(0).sample([r for r in rows if r["id"] in set(off_train)], len(off_valid))]
    settings = {
        "공식 분할 검증 (후보 800, 스타터 키트와 같음)": sibling_stats(off_valid, off_valid, group_of),
        "기존 로컬 검증 1:2 (후보 1,600 = 검증 800 + 학습 800)": sibling_stats(off_valid, off_valid + extra, group_of),
        "그룹 분할 검증 (검증 이미지 전체)": sibling_stats(valid, valid, group_of),
    }

    # 4) 샘플러 점검 (그룹 분할 학습 ID, 에포크 0)
    clear_pairs = [(p["a"], p["b"]) for p in pairs if p["clear"]]
    tr = set(train)
    clear_train = [(a, b) for a, b in clear_pairs if a in tr and b in tr]
    rng_a, rng_b = random.Random(seed), random.Random(seed)
    A = random_batches(train, bs, rng_a)
    B, chosen = sibling_batches(train, clear_train, bs, pairs_per_batch, rng_b)
    sampler = {"bs": bs, "pairs_per_batch": pairs_per_batch, "clear_pairs_in_train": len(clear_train),
               "chosen_pairs_epoch0": len(chosen),
               "same_ids_and_exposure": sorted(x for b in A for x in b) == sorted(x for b in B for x in b),
               "same_batch_sizes": sorted(map(len, A)) == sorted(map(len, B)),
               "A": batch_pair_stats(A, group_of, pair_index), "B": batch_pair_stats(B, group_of, pair_index),
               "valid_ids_in_batches": len({x for b in A + B for x in b} & set(valid))}

    # 5) 집계
    def tally(sel):
        c = Counter()
        for p in sel:
            for cat in {d[0] for d in p["diffs"] if d[1] == "explicit"}:
                c[cat] += 1
        return dict(c.most_common())
    single = Counter(p["object_categories"][0] for p in pairs if p["n_explicit"] == 1 and p["n_explicit_object"] == 1)
    flags = Counter(f.split(":")[0] for v in parsed.values() for f in v["flags"])
    unknown = Counter(f.split(":")[1] for v in parsed.values() for f in v["flags"] if f.startswith("unknown_noun"))
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parse": {"n": len(rows), "with_objects": sum(bool(v["objects"]) for v in parsed.values()),
                  "objects_per_prompt": round(sum(len(v["objects"]) for v in parsed.values()) / len(rows), 2),
                  "relations_per_prompt": round(sum(len(v["relations"]) for v in parsed.values()) / len(rows), 2),
                  "prompts_with_flags": sum(bool(v["flags"]) for v in parsed.values()),
                  "flag_counts": dict(flags.most_common()), "unknown_nouns": dict(unknown.most_common(15))},
        "groups": {"n_groups": len(members), "size_dist": dict(sorted(Counter(map(len, members.values())).items())),
                   "n_with_siblings": sum(len(v) for v in members.values() if len(v) > 1)},
        "pairs": {"n_pairs": len(pairs), "diff_class": dict(Counter(p["diff_class"] for p in pairs)),
                  "clear": sum(p["clear"] for p in pairs),
                  "explicit_by_category_overlapping": tally(pairs),
                  "single_explicit_object_diff": dict(single.most_common()),
                  "missing_by_category": dict(Counter(d[0] for p in pairs for d in p["diffs"] if d[1] == "missing").most_common())},
        "legacy": legacy_counts(rows),
        "group_split": {k: v for k, v in split.items() if k not in ("train_ids", "valid_ids", "group_of")},
        "validation_settings": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in settings.items()},
        "sampler_check": sampler,
    }
    summary["examples"] = save_examples(pairs, data_dir / "images", report_dir / "examples.jpg")

    # 6) 대조 문구 (그룹 분할 학습 쌍만, 판별력 점검 전용) + 점검 세트
    phrase_rows = []
    for p in pairs:
        if p["a"] in tr and p["b"] in tr and (p["single_status"] == "단일 확정" or p["binding_swap"]):
            for ph in contrast_phrases(parsed[p["a"]], parsed[p["b"]], p["diffs"], p["a"], p["b"]):
                phrase_rows.append({"group": p["group"], "id_a": p["a"], "id_b": p["b"], **ph})
    with (report_dir / "contrast_phrases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "id_a", "id_b", "category", "target_image", "other_image",
                                          "target", "contrast", "use"])
        w.writeheader()
        w.writerows(phrase_rows)
    probe = build_probe_set(pairs, phrase_rows, tr, seed, checked=checks)
    with (report_dir / "probe_set.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n", "probe_category", "group", "id_a", "id_b", "prompt_a", "prompt_b", "single_status",
                    "text_result", "binding_swap", "target(id_a에 맞음)", "contrast(id_b에 맞음)",
                    "image_check", "image_check_note", "checked_by"])
        for k, (pc, p, ph) in enumerate(probe, 1):
            chk = checks.get((p["a"], p["b"]), ("", "", ""))
            w.writerow([k, pc, p["group"], p["a"], p["b"], p["prompt_a"].replace("\n", " "), p["prompt_b"].replace("\n", " "),
                        p["single_status"], fmt_diffs(p["diffs"], "explicit"), p["binding_swap"],
                        ph["target"], ph["contrast"], chk[0], chk[1], chk[2]])
    save_probe_sheet(probe, data_dir / "images", report_dir / "probe_set.jpg")
    summary["probe"] = {
        "scope": "그룹 분할 학습 ID 안의 쌍만 (최종 검증 세트는 쓰지 않음)",
        "single_status_all_pairs": dict(Counter(p["single_status"] for p in pairs)),
        "single_confirmed_in_train_by_category": dict(Counter(
            p["object_categories"][0] for p in pairs if p["single_status"] == "단일 확정" and p["a"] in tr and p["b"] in tr).most_common()),
        "binding_swaps_all": dict(Counter(k.split(":")[0] for p in pairs for k in p["binding_swap"].split(" | ") if k)),
        "binding_swaps_in_train": dict(Counter(k.split(":")[0] for p in pairs if p["a"] in tr and p["b"] in tr for k in p["binding_swap"].split(" | ") if k)),
        "phrase_rows_by_category": dict(Counter(r["category"] for r in phrase_rows).most_common()),
        "probe_set_by_category": dict(Counter(pc for pc, _, _ in probe).most_common()),
        "manual_checks_loaded": len(checks),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (report_dir / "summary.md").write_text(render_md(summary), encoding="utf-8")
    return summary


PROBE_CATS = [("color", "색"), ("count", "개수"), ("count_binding_swap", "개수 결합 교차"), ("state", "상태"),
              ("relation_left_right", "좌우 관계"), ("color_binding_swap", "색 결합 교차")]


def build_probe_set(pairs, phrase_rows, train_ids, seed, per_cat=6, checked=()):
    """학습 분할 안에서 조건별 대표 쌍. 대조 문구가 만들어진 실제 쌍만 쓴다.
    이미 이미지를 확인한 쌍(checked)을 먼저 넣어, 다시 실행해도 점검 세트가 크게 바뀌지 않게 한다."""
    by_pair = defaultdict(list)
    for r in phrase_rows:
        by_pair[(r["id_a"], r["id_b"])].append(r)
    index = {(p["a"], p["b"]): p for p in pairs}
    rng = random.Random(seed)
    checked = set(checked)
    out = []
    for cat, _ in PROBE_CATS:
        pool = [(index[k], r) for k, rs in by_pair.items() for r in rs if r["category"] == cat]
        pool.sort(key=lambda x: (x[0]["a"], x[0]["b"]))
        first = [x for x in pool if (x[0]["a"], x[0]["b"]) in checked][:per_cat]
        rest = [x for x in pool if (x[0]["a"], x[0]["b"]) not in checked]
        out += [(cat, p, r) for p, r in first + rng.sample(rest, min(per_cat - len(first), len(rest)))]
    return out


def save_probe_sheet(probe, images_dir, path, thumb=224):
    from PIL import Image, ImageDraw
    if not probe:
        return
    pad, text_h = 8, 110
    W, H = 2 * thumb + 3 * pad, len(probe) * (thumb + text_h + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for n, (cat, p, r) in enumerate(probe, 1):
        for k, key in enumerate(("a", "b")):
            with Image.open(Path(images_dir) / f"{p[key]}.png") as im:
                sheet.paste(im.convert("RGB").resize((thumb, thumb)), (pad + k * (thumb + pad), y))
        lines = [f"#{n} [{cat}] {p['a']}(left) vs {p['b']}(right) {p['group']}",
                 f"target(left): {r['target']}", f"contrast(right): {r['contrast']}"]
        for key in ("a", "b"):
            t = p[f"prompt_{key}"].replace("\n", " ")
            lines += [f"{p[key]}: {t[:72]}", f"   {t[72:144]}"]
        for k, line in enumerate(lines):
            draw.text((pad, y + thumb + 3 + k * 14), line, fill="black")
        y += thumb + text_h + pad
    sheet.save(path, quality=85)


def render_md(s):
    L = []
    a = L.append
    lg, pr, gp, sp = s["legacy"], s["pairs"], s["group_split"], s["sampler_check"]
    a("# 형제 그룹 분석과 그룹 분할\n")
    a(f"- 생성: {s['generated_at']} · 원본은 읽기만 함 · 판정은 **텍스트 기준 규칙 파서**(근사치)")
    a("- 파일: `pairs.csv`(쌍 표), `parsed.jsonl`(설명별 파싱 결과), `examples.jpg`(대표 쌍 이미지), `../../splits/split_group_seed42.json`\n")
    a("## 1. 기존 수치(관계 1,137 / 색 713) 재현\n")
    a(f"- 범위: {lg['scope']}. 그룹 {lg['n_groups']}개, 형제가 있는 설명 {lg['n_with_siblings']}개, 쌍 {lg['n_pairs']}개")
    a(f"- 쌍 세는 법: {lg['pair_counting']}")
    a(f"- **한 범주만 다른 쌍 (서로 겹치지 않음)**: {lg['single_category_exclusive']}")
    a(f"- 범주별 쌍 수 (한 쌍이 여러 범주에 동시에 들어감): {lg['category_overlapping']}")
    a(f"- 내용 단어 집합이 완전히 같은 쌍: {lg['no_content_difference']}")
    a("- 주의: 이 방식의 '관계'는 **관계 단어 집합이 다르다**는 뜻이다(예: above ↔ next to). 방향만 바뀐 쌍은 단어가 같아서 '차이 없음'으로 들어간다. "
      "'색'도 어느 물체의 색인지 보지 않는다. 또 그룹 키에서 개수 단어를 뺐기 때문에 개수가 다른 설명도 같은 그룹이다.\n")
    a("## 2. 새 정의와 파싱 품질\n")
    p = s["parse"]
    a("- 물체 = 사전에 있는 물체 이름(여러 단어 포함). 개수는 소개 문구(one/two/three/single)에서 따로 센다.")
    a("- 형제 그룹 키 = 물체 **종류** 집합(개수 제외). 개수 차이는 `count` 조건으로 기록한다.")
    a("- 속성은 물체에 연결해 비교한다(예: `색[cat] black→brown`). 관계는 정규화한다(right_of→left_of 뒤집기, below/under→above 뒤집기, contains→inside).")
    a(f"- 파싱: 설명 {p['n']}개 중 물체를 찾은 설명 {p['with_objects']}개, 설명당 물체 {p['objects_per_prompt']}개, 관계 {p['relations_per_prompt']}개")
    a(f"- 애매 표시가 있는 설명: {p['prompts_with_flags']}개 — {p['flag_counts']}")
    a(f"- 사전에 없는 물체 후보 상위: {p['unknown_nouns']}\n")
    g = s["groups"]
    a(f"그룹 {g['n_groups']}개, 형제가 있는 설명 {g['n_with_siblings']}개, 그룹 크기 분포 {g['size_dist']}\n")
    a("## 3. 같은 물체 집합 안의 차이\n")
    a(f"- 쌍 {pr['n_pairs']}개 (같은 그룹 안 A<B 한 번씩, 4,000쌍 전체 기준). 차이 수: {pr['diff_class']}")
    a(f"- 샘플러에 쓸 수 있는 명확한 쌍(물체 조건이 명시적으로 다르고 파싱 애매 없음): **{pr['clear']}**")
    a("\n| 조건 | 명시적으로 다른 쌍 (겹침 허용) | 이 조건 하나만 다른 쌍 |")
    a("|---|---|---|")
    for cat in OBJECT_CATS + SCENE_CATS:
        a(f"| {CAT_KO[cat]} | {pr['explicit_by_category_overlapping'].get(cat, 0)} | {pr['single_explicit_object_diff'].get(cat, '—') if cat in OBJECT_CATS else '—'} |")
    a(f"\n- 한쪽에만 명시된 조건(누락) 쌍 수: {pr['missing_by_category']}")
    a("- '이 조건 하나만 다른 쌍' = 명시적 차이가 전체(장면 포함) 1개이고 그것이 이 물체 조건인 쌍.\n")
    a("## 4. 검증 설정 비교 (모델 없이 계산한 후보 구성)\n")
    a("| 설정 | 질의 | 후보 | 질의당 형제 오답 평균 | 형제 오답이 1장 이상인 질의 |")
    a("|---|---|---|---|---|")
    for k, v in s["validation_settings"].items():
        a(f"| {k} | {v['n_queries']} | {v['n_gallery']} | {v['mean_siblings_per_query']} | {v['queries_with_sibling']} ({v['ratio_with_sibling']:.1%}) |")
    a("\n- **기존 로컬 검증 1:2**는 팀이 01 노트북에서 만든 설정이다: 공식 검증 질의 800개, 후보 = 검증 이미지 800 + 학습 이미지 무작위 800(`random.Random(0)`). "
      "추가된 800장은 학습된 모델이 이미 본 이미지라는 점을 감안해야 한다.")
    a(f"- 그룹 분할: 학습 {gp['n_train']} / 검증 {gp['n_valid']}, 그룹 {gp['n_groups_train']} / {gp['n_groups_valid']}. 누수 점검 {gp['leak_check']}")
    a("- 그룹 분할 검증 점수는 **처음 보는 물체 조합에 대한 일반화**도 함께 잰다. 공식 분할과의 점수 차이를 모두 형제 오답 때문이라고 해석하지 않는다.")
    a("- 비공개 Test와의 유사도(추정): Test 후보는 16,000장이고 같은 방식으로 생성됐다면 질의당 형제 오답은 train보다 많을 것이다. "
      "Test의 물체 조합이 train과 얼마나 겹치는지는 알 수 없다. 그룹 분할은 '겹치지 않는' 쪽의 극단, 공식 분할은 '형제가 적은' 쪽의 극단으로 본다.\n")
    a("## 5. 샘플러 점검 (그룹 분할 학습 ID, 에포크 0, 실제 배치 기준)\n")
    a(f"- 배치 크기 {sp['bs']}, B의 배치당 비교 쌍 최대 {sp['pairs_per_batch']}개. 학습 분할 안 명확한 쌍 {sp['clear_pairs_in_train']}개 중 이번 에포크 선택 {sp['chosen_pairs_epoch0']}개")
    a(f"- A/B 같은 ID 집합·노출 횟수: {sp['same_ids_and_exposure']}, 같은 배치 크기 구성: {sp['same_batch_sizes']}, 배치에 들어간 검증 ID: {sp['valid_ids_in_batches']}")
    a("\n| 배치 | 배치 수 | 같은 배치의 형제 쌍 | 그중 명확한 쌍 | 배치당 명확한 쌍 | 조건 종류 |")
    a("|---|---|---|---|---|---|")
    for k in ("A", "B"):
        v = sp[k]
        a(f"| {k} | {v['n_batches']} | {v['sibling_pairs_in_batches']} | {v['clear_pairs_in_batches']} | {v['clear_pairs_per_batch']} | {v['categories']} |")
    a("\n## 6. 대표 쌍 (`examples.jpg`)\n")
    a("| # | 조건 | A | B | 그룹 | 명시적 차이 |")
    a("|---|---|---|---|---|---|")
    for e in s["examples"]:
        a(f"| {e['n']} | {e['category']} | {e['a']} | {e['b']} | {e['group']} | {e['explicit']} |")
    a("\n이미지 확인 결과는 `manual_checks.csv`에 판정자(`checked_by`)와 함께 적고, 실행할 때 `pairs.csv`·`probe_set.csv`의 `image_check` 열로 합친다. 현재 판정은 모두 Claude가 이미지를 보고 한 것이며(`checked_by = Claude`), 팀원 검토는 아직 하지 않았다.")
    pb = s.get("probe")
    if pb:
        a("\n## 7. 단일 속성 쌍, 점검 세트, 대조 문구 (후속 재정렬 실험용 전처리)\n")
        a(f"- 범위: {pb['scope']}")
        a(f"- 단일 속성 판정(전체 쌍): {pb['single_status_all_pairs']}")
        a("  - `단일 확정`: 명시적 차이가 물체 조건 1개뿐이고, 한쪽에만 적힌 조건·동시에 참일 수 있는 관계·파싱 애매가 없음")
        a("  - `단일(미확인)`: 명시적 차이는 1개지만 다른 조건이 한쪽에서 생략되어 나머지가 같은지 확인할 수 없음")
        a(f"- 학습 분할 안 `단일 확정` 쌍 (조건별): {pb['single_confirmed_in_train_by_category']}")
        a(f"- 결합 교차 쌍 (두 물체의 색 또는 개수가 맞바뀜, 실제 데이터에서 발견. 한 쌍이 여러 번 셀 수 있음): 전체 {pb['binding_swaps_all']}, 학습 분할 안 {pb['binding_swaps_in_train']}")
        a(f"- 대조 문구 행 (`contrast_phrases.csv`): {pb['phrase_rows_by_category']}")
        a(f"- 점검 세트 (`probe_set.csv`, `probe_set.jpg`): {pb['probe_set_by_category']}")
        a("- 대조 문구는 **판별력 점검 전용**이다(`use = probe_only`). 정답 캡션이나 학습 데이터로 저장·사용하지 않는다. "
          "두 실제 설명의 값으로만 만들고, 한 물체에 값이 여러 개이거나 같은 종류 물체가 여럿이거나 관계가 left/above 외인 경우는 건너뛴다.")
        a("- 이후 사용 규칙: 판별 비율은 학습 분할 안 개발용 쌍으로만 잰다. 최종 검증 세트는 조건 선택 기준을 정하는 데 쓰지 않는다. "
          "검색 시 각 질의의 순위는 그 질의와 후보 이미지로만 계산하고, 후보 이미지의 짝 설명은 쓰지 않는다.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("haikuthon-train"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/siblings"))
    ap.add_argument("--split-dir", type=Path, default=Path("splits"))
    ap.add_argument("--official-split", type=Path, default=Path("splits/split_seed42.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    s = run(args.data_dir, args.report_dir, args.split_dir, args.official_split, args.seed)
    print(json.dumps({k: s[k] for k in ("parse", "groups", "pairs", "group_split")}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(s["validation_settings"], ensure_ascii=False, indent=1))
    print(json.dumps(s["sampler_check"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
