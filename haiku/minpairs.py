"""최소 쌍(한 가지만 다른 쌍) 만들기: BLIP 학습용 어려운 오답 (NegCLIP 계열, 규칙 기반 — 생성 모델 사용 안 함).

- 문장 쪽: 정답 문장에서 한 가지만 바꾼 문장을 오답으로 쓴다.
  ① 속성 주인 바꾸기 (같은 종류 속성 두 값을 맞바꿈: "빨간 컵·파란 접시" → "파란 컵·빨간 접시")
  ② 속성 값 바꾸기 (문장에 없는 다른 값으로: 빨강 → 초록)
  ③ 방향 뒤집기 (왼쪽 ↔ 오른쪽, 위 ↔ 아래)
  ④ 상태 뒤집기 (열림 ↔ 닫힘 등)
- 이미지 쪽: 좌우가 나오는 문장은 좌우로 뒤집은 정답 이미지를 오답으로 쓴다 (나머지는 모두 같고 방향만 반대).
학습 데이터 문장과 이미지만 쓴다.
"""

import random
import re

FAMILIES = {
    "color": "red orange yellow green blue purple pink brown black white gray beige".split(),
    "pattern": "striped polka-dotted checkered floral zigzag-patterned solid-colored".split(),
    "material": "metal wooden plastic ceramic glass leather fabric wicker rubber".split(),
    "shape": "round square rectangular triangular oval star-shaped hexagonal".split(),
    "pose": "standing sitting lying crouching kneeling".split(),
}
FLIPS = [("left", "right"), ("above", "below"), ("open", "closed"), ("lit", "unlit"), ("folded", "unfolded"),
         ("inflated", "deflated"), ("empty", "full")]
_W = r"(?<![\w-]){}(?![\w-])"
_SKIP = [re.compile(r"orange (?:fruit|fruits)\b"), re.compile(r"drinking glass"), re.compile(r"\bglass(?:es)? of\b")]
_LR = re.compile(r"(?<![\w-])(left|right)(?![\w-])", re.I)


def _norm(s):
    return s.replace("grey", "gray").replace("Grey", "Gray")


def _protected(text):
    spans = []
    for p in _SKIP:
        spans += [m.span() for m in p.finditer(text.lower())]
    return spans


def _sub_words(text, mapping):
    """단어 경계에서 mapping의 단어를 한 번에 바꾼다 (보호 구간 제외, 첫 글자 대소문자 유지)."""
    prot = _protected(text)
    pat = re.compile(_W.format("(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + ")"), re.I)

    def f(m):
        if any(a <= m.start() < b for a, b in prot):
            return m.group(0)
        w = m.group(0)
        r = mapping[w.lower()]
        return r[:1].upper() + r[1:] if w[:1].isupper() else r
    return pat.sub(f, text)


def _present(text, words):
    low = text.lower()
    prot = _protected(text)
    out = []
    for w in words:
        for m in re.finditer(_W.format(re.escape(w)), low):
            if not any(a <= m.start() < b for a, b in prot):
                out.append(w)
                break
    return out


def perturb(prompt, rng, n=4):
    """정답 문장 → 한 가지만 다른 오답 문장 최대 n개 (서로 다르고 원문과도 다름)."""
    text = _norm(prompt)
    cands = []
    for fam, words in FAMILIES.items():
        have = _present(text, words)
        if len(have) >= 2:                                               # ① 주인 바꾸기
            a, b = rng.sample(have, 2)
            cands.append(("swap_" + fam, _sub_words(text, {a: b, b: a})))
        if have:                                                         # ② 값 바꾸기
            a = rng.choice(have)
            others = [w for w in words if w not in have]
            if others:
                cands.append(("replace_" + fam, _sub_words(text, {a: rng.choice(others)})))
    for a, b in FLIPS:                                                   # ③④ 방향·상태 뒤집기
        if _present(text, [a]) or _present(text, [b]):
            cands.append(("flip_" + a, _sub_words(text, {a: b, b: a})))
    rng.shuffle(cands)
    out, seen = [], {text.lower()}
    for kind, t in cands:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) >= n:
            break
    return out


def has_left_right(prompt):
    return bool(_LR.search(prompt))


if __name__ == "__main__":
    rng = random.Random(0)
    for p in ["One red cup is to the left of one blue plate. The cup is open.",
              "A watercolor shows one striped wooden box above one orange fruit and one drinking glass, standing."]:
        print(p)
        for t in perturb(p, rng, 6):
            print("   ->", t)
