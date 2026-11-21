# Train 데이터 점검 보고서

- 생성: 2026-11-21T16:58:21 · 실행 시간 14.4초 (이미지 디코딩 13.0초, worker 16)
- 데이터: `C:\데이터톤\haikuthon-train`
- 데이터 지문(SHA-256): `9df6a2aa058d58aa…` — 분할 파일과 같은 데이터인지 확인할 때 쓴다
- 원본 파일은 읽기만 했다. 삭제·수정한 파일은 없다.

## 1. 요약

| 점검 | 결과 |
|---|---|
| 텍스트 행 / 고유 ID | 4000 / 4000 |
| ID 형식 | 네 자리 문자열 예, 연속 예, 범위 ['0000', '3999'] |
| 중복 ID | 0 |
| 누락 이미지 / 설명 없는 이미지 | 0 / 0 |
| 빈 설명 | 0 |
| 이미지 디코딩 성공 / 손상 | 4000 / 0 |
| 이미지 형식 · 모드 · 크기 | {'PNG': 4000} · {'RGB': 4000} · {'768x768': 4000} |
| 완전 중복 `image_file_sha256` | 0개 그룹 |
| 완전 중복 `image_pixels_md5` | 0개 그룹 |
| 완전 중복 `text_raw` | 0개 그룹 |
| 완전 중복 `text_normalized` | 0개 그룹 |
| 완전 중복 `image_and_text_pair` | 0개 그룹 |
| 확정 중복 그룹 (분할 때 같은 쪽에 묶음) | 0개 |
| 이미지 근접 중복 후보 (pHash ≤ 8) | 115쌍 |
| 텍스트 유사 쌍 (Jaccard ≥ 0.9) | 295쌍 |

## 2. 실제 디렉터리 구조

- 최상위: `.cache/`, `.gitattributes`, `README.md`, `assets/`, `images/`, `prompts.jsonl`, `starterkit.zip`
- `images/` 확장자: {'.png': 4000, '.jsonl': 1}
- `images/`의 PNG 외 파일: ['metadata.jsonl']
- `images/metadata.jsonl`: 4000행, prompts.jsonl과 일치 예
- `prompts.jsonl` 형식: 한 줄에 `{"id": "0000", "prompts": "..."}`. ID는 문자열로 읽어서 앞자리 0을 유지한다.

## 3. 텍스트 통계

| 항목 | 최소 | 평균 | 중앙값 | 95% | 최대 |
|---|---|---|---|---|---|
| 글자 수 | 104 | 198.39 | 195 | 266 | 378 |
| 단어 수 | 17 | 36.44 | 36 | 49 | 72 |
| 문장 수 | 1 | 2.57 | 2 | 4 | 8 |

## 4. 이미지 통계

- 파일 크기(바이트): 최소 423158, 중앙값 908552, 최대 1604385
- 픽셀 표준편차: 최소 13.51, 중앙값 53.36, 최대 94.26 (거의 단색 기준 < 3.0)

## 5. 중복과 유사 데이터

### 완전 중복 (확정)

| 종류 | 그룹 수 | 예시 |
|---|---|---|
| `image_file_sha256` | 0 | — |
| `image_pixels_md5` | 0 | — |
| `text_raw` | 0 | — |
| `text_normalized` | 0 | — |
| `image_and_text_pair` | 0 | — |

### 이미지 근접 중복 후보 (확정 아님)

- 지각 해시는 구도·배치가 비슷하면 가깝게 나온다. 색·재질처럼 속성 하나만 다른 이미지도 후보에 들어오므로 후보는 반드시 `near_dup_image_pairs.png`로 육안 확인한다.
- 방법: pHash(32x32 DCT 저주파 8x8, 64비트) 해밍 거리, 후보 쌍에는 dHash 거리도 함께 기록
- 각 이미지의 가장 가까운 다른 이미지까지 pHash 거리 분포: {'0': 0, '1-4': 12, '5-8': 154, '9-12': 597, '13-16': 2317, '17-20': 920, '>20': 0}
- 후보 115쌍, 관련 이미지 166장
- 상위 쌍 (육안 확인용 그림: `near_dup_image_pairs.png`):

| A | B | pHash | dHash |
|---|---|---|---|
| 2348 | 2539 | 4 | 4 |
| 0146 | 2675 | 4 | 7 |
| 0696 | 2153 | 4 | 7 |
| 0739 | 2256 | 4 | 8 |
| 2273 | 2348 | 4 | 8 |
| 0101 | 0623 | 4 | 11 |
| 1994 | 2256 | 4 | 11 |
| 1933 | 2177 | 6 | 1 |
| 2375 | 3222 | 6 | 4 |
| 1076 | 3893 | 6 | 5 |
- **검사하지 않은 것**: 의미 수준(임베딩) 유사도는 검사하지 않음. 같은 장면을 다른 구도로 그린 이미지는 해시로 잡히지 않는다.

### 텍스트 유사

- 방법: 단어 집합 Jaccard (소문자, 영숫자 단어), 어휘 587개
- 각 문장의 가장 비슷한 다른 문장까지 Jaccard 분포: {'1.0': 10, '0.9-1.0': 476, '0.8-0.9': 1759, '0.7-0.8': 1404, '<0.7': 351}
- Jaccard ≥ 0.9 쌍: 295개
- 같은 단어 집합(순서·결합만 다름) 그룹: 2개
- 주의: 단어 집합이 같아도 속성이 다른 객체에 붙거나 관계 방향이 바뀌면 다른 장면이다. 이 지표는 중복 판정에 쓰지 않는다.

  - `0005` vs `0333` (Jaccard 1.0)
    - 0005: One horse, lying above one fox, is in a meadow. The scene is a pencil sketch shown from the front in direct sunlight.
    - 0333: In a meadow, one fox is above one lying horse. The scene is a pencil sketch shown from the front in direct sunlight.

  - `0339` vs `3393` (Jaccard 1.0)
    - 0339: A photograph from a front view shows one picture frame, oval, blue, and plastic, one sunflower, and one rectangular block made of metal in a studio with a plain backdrop under warm indoor lighting. The sunflower is to the right of the picture frame, and the rectangular block is below the picture frame.
    - 3393: A photograph from a front view shows one picture frame, oval, blue, and made of metal, in a studio with a plain backdrop under warm indoor lighting. One sunflower is to the right of the picture frame, and one rectangular block made of plastic is below the picture frame.

  - `0849` vs `1958` (Jaccard 1.0)
    - 0849: One pen is inside one bowl, purple and striped, and next to one suitcase, open, yellow, and checkered. One camera is blue. The scene is in a studio with a plain backdrop, rendered in low-poly 3D from an overhead view with warm indoor lighting.
    - 1958: One pen is inside one bowl, yellow and striped, and next to one suitcase, open, purple, and checkered. One camera is blue. The scene is in a studio with a plain backdrop, rendered in low-poly 3D from an overhead view with warm indoor lighting.

## 6. 발견한 문제 목록

| 수준 | 점검 | 내용 | 개수 | ID 예시 |
|---|---|---|---|---|
| 참고 | 비ASCII 문자 | 영문 외 문자가 들어 있음 | 1 | 1732 |
| 참고 | 문단 줄바꿈 | 설명 안에 줄바꿈(\n\n)이 있음. 문단 구분으로 보이며 원본 유지 | 22 | 0013, 0276, 0363, 0425, 0735, 0841, 0986, 1334 |
| 참고 | 같은 단어 집합 텍스트 | 2개 그룹. 속성·관계 결합이 달라 중복으로 확정하지 않음 | 4 | 0005, 0333, 0849, 1958 |

## 7. 이번 점검의 범위

- 수행: ID 대응, 앞자리 0 유지, 중복 ID, 빈 설명, 누락·짝 없는 이미지, 전체 디코딩(손상), 형식·모드·크기 분포, 파일·픽셀·텍스트 완전 중복, pHash/dHash 이미지 근접 중복 후보, 단어 Jaccard 텍스트 유사도
- 수행하지 않음: 임베딩(CLIP/SigLIP) 기반 의미 유사 이미지 탐지, 이미지와 설명의 내용 일치 여부 검사
- 원본 텍스트·이미지는 수정하지 않았다.
