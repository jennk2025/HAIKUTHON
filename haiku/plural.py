"""복수 주체 공간 관계 점검: "두 개의 X가 한 개의 Y 위/아래/왼쪽/오른쪽"처럼 같은 종류 두 개가 기준 하나와 관계를 맺는 문장.

- 문장 파서와 박스 판정 규칙은 팀원의 원 실험(ROI Plural Top2, Grounding DINO)에서 그대로 가져왔다.
  검출기만 기존 OWL-ViT(여러 상자 검출)로 바꿔 추가 모델과 시간 부담 없이 쓴다.
- 판정: 주체 상자가 정확히 2개, 기준 상자가 정확히 1개일 때만. 두 주체 모두 방향을 만족하면 satisfy,
  하나라도 반대면 contradict, 나머지·검출 부족·과다는 abstain. 1위 contradict + 2위 satisfy일 때만 교환.
- 현재 질의 문장과 그 후보 이미지만 쓴다.
"""

import math
import re
from itertools import combinations

# ---------------------------------------------------------------- 원 실험의 명사·속성·방향 정의 (팀원 코드)
ROI_NOUNS = tuple(x.strip() for x in """
acoustic guitar|airplane|apple|armchair|backpack|ball|banana|baseball bat|basketball|bear|bed|bench|bicycle|
bird|block|boat|bonsai tree|book|boot|bottle|bowl|box|broccoli|bus|cabinet|cactus|camera|car|carrot|cat|
chair|chicken|clipboard|clock|coat rack|coffee table|colander|cone|cookie|cow|croissant|cube|cucumber|cupcake|
cushion|cylinder|deer|desk lamp|desk|dog|dolphin|donkey|donut|dress|drinking glass|duck|dumbbell|
ear of corn|egg|elephant|envelope|eraser|fern|flower pot|food container|fork|fox|frying pan|gardening trowel|
giraffe|glove|goat|guinea pig|hammer|hamster|hand saw|handbag|hat|headphones|helicopter|helmet|horse|jacket|
kettle|kick scooter|kite|knife|laptop computer|lemon|loaf of bread|motorcycle|mug|notebook|orange fruit|owl|
paint roller|paintbrush|palm tree|parrot|pear|pen|pencil case|pencil|penguin|person|picture frame|pig|
pillow|pine tree|pitcher|plate|pliers|pyramid|pumpkin|rabbit|rake|robot toy|roller skate|rose|rug|ruler|
sandwich|saucepan|scarf|scissors|screwdriver|sheep|shelf|shirt|shopping cart|shovel|skateboard|ski|sleeping bag|
slice of pizza|small boat|smartphone|sneaker|snowboard|soccer ball|sock|sofa|spoon|stapler|star-shaped block|
stool|strawberry|suitcase|sunflower|sunglasses|surfboard|table knife|table|tablet device|tape measure|
teddy bear|tennis ball|tennis racket|tent|teapot|tomato|toy car|toy train|train|tray|truck|tulip|turtle|
umbrella|vase|violin bow|violin|wall clock|watering can|wedge of cheese|wheelbarrow|wrench|wristwatch|zebra
""".replace("\n", "").split("|") if x.strip())

ROI_ALIASES = {x: x for x in ROI_NOUNS}
ROI_ALIASES.update({
    'ski boot': 'boot', 'computer tablet': 'tablet device', 'mobile phone': 'smartphone', 'trowel': 'gardening trowel',
    'guitar': 'acoustic guitar', 'racket': 'tennis racket', 'scooter': 'kick scooter', 'tablet': 'tablet device',
    'laptop': 'laptop computer', 'boat': 'boat', 'saw': 'hand saw', 'lamp': 'lamp', 'motorbike': 'motorcycle',
})
_ROI_ATTRS = set("""red orange yellow green blue purple pink brown black white gray beige striped polka-dotted checkered
floral zigzag-patterned solid-colored round square rectangular triangular oval star-shaped metal wooden plastic ceramic
glass leather fabric wicker rubber open closed""".split())
_ROI_COLOR = set("red orange yellow green blue purple pink brown black white gray beige".split())
_ROI_PATTERN = set("striped polka-dotted checkered floral zigzag-patterned solid-colored".split())
_ROI_SHAPE = set("round square rectangular triangular oval star-shaped".split())
_ROI_REL = re.compile(r"\b(?:(?:to (?:the )?|on the |the )?(left|right) of|(above|below))\b")
_ROI_SENTENCE = re.compile(r"[.!?;\n]+")
_ROI_PLURAL_FORMS = {
    'person': 'people', 'knife': 'knives', 'table knife': 'table knives', 'scarf': 'scarves', 'shelf': 'shelves',
    'loaf of bread': 'loaves of bread', 'ear of corn': 'ears of corn', 'slice of pizza': 'slices of pizza',
    'wedge of cheese': 'wedges of cheese', 'cactus': 'cacti', 'bus': 'buses', 'fox': 'foxes',
}


def _roi_plural(word):
    if word in _ROI_PLURAL_FORMS:
        return _ROI_PLURAL_FORMS[word]
    if word in {'sheep', 'deer', 'broccoli', 'sunglasses', 'scissors', 'pliers', 'headphones', 'ski'}:
        return word if word != 'ski' else 'skis'
    if word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + ('es' if word.endswith(('ch', 'sh', 's', 'x', 'o')) else 's')


_ROI_SINGULAR = dict(ROI_ALIASES)
for _alias, _canonical in tuple(ROI_ALIASES.items()):
    _ROI_SINGULAR[_roi_plural(_alias)] = _canonical
_ROI_NP = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(_ROI_SINGULAR, key=len, reverse=True)) + r")\b")


def _roi_norm(s):
    s = str(s).lower().replace("–", "-").replace("‑", "-").replace("grey", "gray")
    for a, b in [('polka dotted', 'polka-dotted'), ('solid colored', 'solid-colored'), ('chequered', 'checkered')]:
        s = s.replace(a, b)
    return s


# ---------------------------------------------------------------- 주체 두 개와 기준 한 개 추출 (팀원 코드)
_PL_GROUP = {"two", "2", "both"}
_PL_SIMPLE_ATTRS = _ROI_ATTRS | {
    "standing", "sitting", "seated", "lying", "crouching", "kneeling", "folded",
    "unfolded", "lit", "unlit", "inflated", "deflated", "small", "wood", "solid",
}
_PL_ATTR_FAMILIES = (
    _ROI_COLOR, _ROI_PATTERN, _ROI_SHAPE,
    {"standing", "sitting", "seated", "lying", "crouching", "kneeling"},
    {"open", "closed"}, {"lit", "unlit"}, {"folded", "unfolded"},
    {"inflated", "deflated"},
)
_PL_ATTR_ALT = "|".join(re.escape(a) for a in sorted(_PL_SIMPLE_ATTRS | {"single"}, key=len, reverse=True))
_PL_PREFIX = re.compile(
    rf"\b(?:(?:the\s+)?(?P<count>two|2|both)|(?P<article>one|a|an|the))\s+"
    rf"(?P<attrs>(?:(?:{_PL_ATTR_ALT})\s+)*)$"
)
_PL_BRIDGE = re.compile(
    rf"(?:[\s,]+|\b(?:{_PL_ATTR_ALT}|both|all|each|which|that|who|are|were|stand|sit|"
    r"lie|rest|hang|located|positioned|placed|directly|immediately|also|and|made of|in color)\b)*"
)
_PL_SCOPE = re.compile(
    r"\b(?:one of|either|neither|nor|not|respectively|another|other|at least|at most|"
    r"more than|fewer than|less than|only one|out of|between|except|rather than|or)\b"
)
_PL_END_ATTRS = re.compile(
    rf"^\s*(?:,\s*(?:(?:which|that)\s+(?:is|are)\s+)?|\s+(?:(?:which|that)\s+)?(?:is|are)\s+|"
    rf"\s+made of\s+)"
    rf"(?:(?:both|made of)\s+)*(?P<desc>(?:{_PL_ATTR_ALT})"
    rf"(?:(?:(?:,\s*(?:and\s+)?|\s+(?:and\s+)?)(?:made of\s+)?)(?:{_PL_ATTR_ALT}))*)"
    r"(?=,|$|\s+(?:pattern|color|and|in|on|with|is|are|to|above|below)\b)"
)
_PL_GROUP_SURFACES = {_roi_plural(alias) for alias in ROI_ALIASES}
_PL_PATTERN_SUFFIX = re.compile(
    r"^\s+with\s+(?:a\s+)?(?P<pattern>" + "|".join(sorted(_ROI_PATTERN)) + r")\s+pattern\b"
)
_PL_UNCOUNTABLE = {"pliers", "headphones", "sunglasses", "scissors", "broccoli"}
_PL_WORDS = re.compile(r"[a-z]+(?:-[a-z]+)*|\d+")


def _pl_mentions(text):
    mentions = []
    for sent_no, sentence in enumerate(_ROI_SENTENCE.split(text)):
        for m in _ROI_NP.finditer(sentence):
            if m[0] == "train" and re.match(r"\s+(platform|station)\b", sentence[m.end():]):
                continue
            prefix = _PL_PREFIX.search(sentence[:m.start()])
            suffix = _PL_END_ATTRS.match(sentence[m.end():])
            attrs = _PL_WORDS.findall(prefix["attrs"]) if prefix else []
            if suffix:
                attrs += _PL_WORDS.findall(suffix["desc"])
            pattern_suffix = _PL_PATTERN_SUFFIX.match(sentence[m.end():])
            if pattern_suffix:
                attrs.append(pattern_suffix["pattern"])
            mentions.append({
                "sentence": sent_no, "start": m.start(), "end": m.end(), "noun": _ROI_SINGULAR[m[0]],
                "surface": m[0], "prefix_start": prefix.start() if prefix else m.start(),
                "quantity": (prefix["count"] or prefix["article"]) if prefix else None,
                "attrs": [a for a in attrs if a in _PL_SIMPLE_ATTRS],
            })
    return mentions


def _pl_phrase(noun, attrs):
    unique = list(dict.fromkeys("wooden" if x == "wood" else x for x in attrs))
    if any(len(set(unique) & family) > 1 for family in _PL_ATTR_FAMILIES):
        return None
    return " ".join([*unique, noun])


def parse_plural_plan(prompt):
    """A single affirmative directional relation from two same-kind subjects to one anchor."""
    def skip(reason):
        return {"eligible": False, "reason": reason, "relations": []}

    text = _roi_norm(prompt)
    if not re.search(r"\b(?:two|2|both)\b", text):
        return skip("no_pair")
    directions = list(_ROI_REL.finditer(text))
    if len(directions) != 1:
        return skip("requires_one_direction")
    if _PL_SCOPE.search(text):
        return skip("unsupported_quantifier_or_scope")

    mentions = _pl_mentions(text)
    sentences = _ROI_SENTENCE.split(text)
    idx, sentence = next((i, s) for i, s in enumerate(sentences) if _ROI_REL.search(s))
    rel = _ROI_REL.search(sentence)
    local = [m for m in mentions if m["sentence"] == idx]
    left = [m for m in local if m["end"] <= rel.start()]
    right = [m for m in local if m["start"] >= rel.end()]
    if not left or not right:
        return skip("unsupported_direction_order")
    subject, anchor = left[-1], right[0]
    bridge = sentence[subject["end"]:rel.start()]
    if not _PL_BRIDGE.fullmatch(bridge):
        return skip("unsupported_subject_clause")
    if sentence[rel.end():anchor["prefix_start"]].strip() or anchor["quantity"] not in {"one", "a", "an", "the"}:
        return skip("unclear_anchor")
    if subject["surface"] not in _PL_GROUP_SURFACES or subject["quantity"] not in _PL_GROUP | {"the"}:
        return skip("unclear_group")
    if anchor["surface"] not in ROI_ALIASES or subject["noun"] == anchor["noun"]:
        return skip("anchor_not_distinct_singular")
    if subject["noun"] in _PL_UNCOUNTABLE or anchor["noun"] in _PL_UNCOUNTABLE:
        return skip("paired_or_mass_noun")
    anchor_tail = sentence[anchor["end"]:]
    if re.match(r"^\s+(?:and|or)\b", anchor_tail) and len(right) > 1:
        return skip("possible_second_anchor")
    left_context = sentence[:subject["prefix_start"]].rstrip()
    if left_context and not (left_context.endswith(",") or re.search(
        r"\b(?:with|shows?|depicts?|features?|contains?|are|and|alongside)$", left_context,
    )):
        return skip("unclear_subject_binding")
    if re.search(r"\b(?:with|and|alongside)$", left_context):
        nearest_clause = left_context.rsplit(",", 1)[-1]
        if _ROI_NP.search(nearest_clause):
            return skip("ambiguous_coordination")

    subject_mentions = [m for m in mentions if m["noun"] == subject["noun"]]
    anchor_mentions = [m for m in mentions if m["noun"] == anchor["noun"]]
    if (not any(m["quantity"] in _PL_GROUP for m in subject_mentions)
            or any(m["surface"] not in _PL_GROUP_SURFACES or m["quantity"] not in _PL_GROUP | {"the"}
                   for m in subject_mentions)
            or any(m["quantity"] not in {"one", "a", "an", "the"} or m["surface"] not in ROI_ALIASES
                   for m in anchor_mentions)
            or sum(m["quantity"] in {"a", "an", "one"} for m in anchor_mentions) > 1
            or sum(m["quantity"] in {"two", "2"} and not re.match(
                r"the\b", sentences[m["sentence"]][m["prefix_start"]:]
            ) for m in subject_mentions) > 1):
        return skip("ambiguous_references")
    subject_attrs = [a for m in subject_mentions for a in m["attrs"]]
    subject_attrs += [a for a in _PL_WORDS.findall(bridge) if a in _PL_SIMPLE_ATTRS]
    subject_label = _pl_phrase(subject["noun"], subject_attrs)
    anchor_label = _pl_phrase(anchor["noun"], [a for m in anchor_mentions for a in m["attrs"]])
    if subject_label is None or anchor_label is None:
        return skip("conflicting_attributes")
    return {"eligible": True, "reason": "two_to_one", "relations": [{
        "subject": subject_label, "relation": rel[1] or rel[2], "object": anchor_label, "count": 2,
    }]}


# ---------------------------------------------------------------- 서로 다른 상자에 공간 조건 적용 (팀원 코드, 확신도 기준만 인자로)
PLURAL_GEOMETRY_SPEC = {"min_confidence": 0.20, "alternative_lead": 0.05, "duplicate_iou": 0.50,
                        "duplicate_center_distance": 0.025}


def _plural_pick_count(raw, count, min_conf):
    ranked = sorted(raw, key=lambda d: d["score"], reverse=True)
    n_confident = sum(d["score"] >= min_conf for d in ranked)
    if n_confident < count:
        return [], "too_few_boxes"
    if n_confident > count:
        return [], "too_many_boxes"
    if len(ranked) > count:
        lead = ranked[count - 1]["score"] - ranked[count]["score"]
        if lead < PLURAL_GEOMETRY_SPEC["alternative_lead"]:
            return [], "ambiguous_cutoff"
    return ranked[:count], "ok"


def _plural_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _plural_may_be_same(a, b):
    ax, ay = _plural_center(a)
    bx, by = _plural_center(b)
    if max(abs(ax - bx), abs(ay - by)) <= PLURAL_GEOMETRY_SPEC["duplicate_center_distance"]:
        return True
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9) >= PLURAL_GEOMETRY_SPEC["duplicate_iou"]


def plural_geometry(found, subject, direction, reference, *, margin=0.08, min_conf=0.20):
    """Evaluate both distinct subjects against one shared reference in one image."""
    subjects, sub_status = _plural_pick_count(found[subject], 2, min_conf)
    references, ref_status = _plural_pick_count(found[reference], 1, min_conf)
    result = {"verdict": "abstain", "reason": "", "distances": []}
    if sub_status != "ok" or ref_status != "ok":
        result["reason"] = "subject_" + sub_status if sub_status != "ok" else "reference_" + ref_status
        return result
    for a, b in combinations(subjects + references, 2):
        if _plural_may_be_same(a["box"], b["box"]):
            result["reason"] = "overlapping_or_duplicate_instances"
            return result
    rx, ry = _plural_center(references[0]["box"])
    for sb in subjects:
        sx, sy = _plural_center(sb["box"])
        d = {"left": rx - sx, "right": sx - rx, "above": ry - sy, "below": sy - ry}[direction]
        if not math.isfinite(d):
            return result
        result["distances"].append(d)
    worst = min(result["distances"])
    if worst >= margin:
        result.update(verdict="satisfy", reason="both_support")
    elif worst <= -margin:
        result.update(verdict="contradict", reason="at_least_one_contradiction")
    else:
        result["reason"] = "within_spatial_margin"
    return result


def owl_found(multi_boxes):
    """OwlChecker.multi_cache 형식 [(score, cx, cy, w, h)] → 원 코드 형식 [{"box": xyxy, "score"}]."""
    out = []
    for s, cx, cy, w, h in multi_boxes:
        out.append({"box": [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)],
                    "score": float(s)})
    return out
