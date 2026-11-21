# 형제 그룹 분석과 그룹 분할

- 생성: 2026-11-21T17:37:44 · 원본은 읽기만 함 · 판정은 **텍스트 기준 규칙 파서**(근사치)
- 파일: `pairs.csv`(쌍 표), `parsed.jsonl`(설명별 파싱 결과), `examples.jpg`(대표 쌍 이미지), `../../splits/split_group_seed42.json`

## 1. 기존 수치(관계 1,137 / 색 713) 재현

- 범위: 공개 train 4,000쌍 전체 (분할 구분 없음). 그룹 1547개, 형제가 있는 설명 3220개, 쌍 6883개
- 쌍 세는 법: 같은 그룹 안에서 A<B 순서로 한 번만 셈 (A–B와 B–A 중복 없음)
- **한 범주만 다른 쌍 (서로 겹치지 않음)**: {'관계': 1137, '색': 713, '화풍': 271, '자세·상태': 193, '무늬': 189, '재질': 122, '조명': 34, '개수': 34, '모양': 13, '시점': 1}
- 범주별 쌍 수 (한 쌍이 여러 범주에 동시에 들어감): {'관계': 3681, '색': 2226, '화풍': 1664, '자세·상태': 914, '무늬': 716, '재질': 442, '조명': 382, '개수': 219, '시점': 145, '모양': 124, '객체·장소': 84}
- 내용 단어 집합이 완전히 같은 쌍: 1006
- 주의: 이 방식의 '관계'는 **관계 단어 집합이 다르다**는 뜻이다(예: above ↔ next to). 방향만 바뀐 쌍은 단어가 같아서 '차이 없음'으로 들어간다. '색'도 어느 물체의 색인지 보지 않는다. 또 그룹 키에서 개수 단어를 뺐기 때문에 개수가 다른 설명도 같은 그룹이다.

## 2. 새 정의와 파싱 품질

- 물체 = 사전에 있는 물체 이름(여러 단어 포함). 개수는 소개 문구(one/two/three/single)에서 따로 센다.
- 형제 그룹 키 = 물체 **종류** 집합(개수 제외). 개수 차이는 `count` 조건으로 기록한다.
- 속성은 물체에 연결해 비교한다(예: `색[cat] black→brown`). 관계는 정규화한다(right_of→left_of 뒤집기, below/under→above 뒤집기, contains→inside).
- 파싱: 설명 4000개 중 물체를 찾은 설명 4000개, 설명당 물체 2.74개, 관계 1.4개
- 애매 표시가 있는 설명: 677개 — {'multi_instance': 475, 'self_relation': 146, 'unbound_attr': 95, 'rel_no_object': 65, 'unknown_noun': 30, 'count_inferred': 30, 'rel_no_subject': 27}
- 사전에 없는 물체 후보 상위: {'is': 15, 'which': 9, 'colored': 2, 'photograph': 1, 'from': 1, 'appear': 1, 'solid': 1}

그룹 656개, 형제가 있는 설명 3999개, 그룹 크기 분포 {1: 1, 2: 4, 4: 323, 7: 1, 8: 319, 12: 2, 16: 3, 20: 1, 24: 2}

## 3. 같은 물체 집합 안의 차이

- 쌍 12129개 (같은 그룹 안 A<B 한 번씩, 4,000쌍 전체 기준). 차이 수: {'복수': 7978, '단일': 3722, '명시 차이 없음': 429}
- 샘플러에 쓸 수 있는 명확한 쌍(물체 조건이 명시적으로 다르고 파싱 애매 없음): **9185**

| 조건 | 명시적으로 다른 쌍 (겹침 허용) | 이 조건 하나만 다른 쌍 |
|---|---|---|
| 개수 | 1626 | 16 |
| 색 | 7026 | 1204 |
| 무늬 | 1349 | 305 |
| 재질 | 620 | 149 |
| 모양 | 183 | 33 |
| 상태 | 281 | 73 |
| 자세 | 1367 | 346 |
| 크기 | 0 | — |
| 관계 방향 | 3704 | 1129 |
| 관계 종류 | 2542 | 467 |
| 화풍 | 736 | — |
| 시점 | 304 | — |
| 조명 | 560 | — |
| 장소 | 480 | — |

- 한쪽에만 명시된 조건(누락) 쌍 수: {'relation_missing': 3918, 'color': 1174, 'pattern': 1061, 'pose': 712, 'material': 267, 'state': 166, 'shape': 109, 'view': 7}
- '이 조건 하나만 다른 쌍' = 명시적 차이가 전체(장면 포함) 1개이고 그것이 이 물체 조건인 쌍.

## 4. 검증 설정 비교 (모델 없이 계산한 후보 구성)

| 설정 | 질의 | 후보 | 질의당 형제 오답 평균 | 형제 오답이 1장 이상인 질의 |
|---|---|---|---|---|
| 공식 분할 검증 (후보 800, 스타터 키트와 같음) | 800 | 800 | 1.192 | 557 (69.6%) |
| 기존 로컬 검증 1:2 (후보 1,600 = 검증 800 + 학습 800) | 800 | 1600 | 2.373 | 723 (90.4%) |
| 그룹 분할 검증 (검증 이미지 전체) | 800 | 800 | 5.702 | 799 (99.9%) |

- **기존 로컬 검증 1:2**는 팀이 01 노트북에서 만든 설정이다: 공식 검증 질의 800개, 후보 = 검증 이미지 800 + 학습 이미지 무작위 800(`random.Random(0)`). 추가된 800장은 학습된 모델이 이미 본 이미지라는 점을 감안해야 한다.
- 그룹 분할: 학습 3200 / 검증 800, 그룹 522 / 134. 누수 점검 {'overlap_ids': 0, 'groups_crossing': 0, 'crossing_examples': []}
- 그룹 분할 검증 점수는 **처음 보는 물체 조합에 대한 일반화**도 함께 잰다. 공식 분할과의 점수 차이를 모두 형제 오답 때문이라고 해석하지 않는다.
- 비공개 Test와의 유사도(추정): Test 후보는 16,000장이고 같은 방식으로 생성됐다면 질의당 형제 오답은 train보다 많을 것이다. Test의 물체 조합이 train과 얼마나 겹치는지는 알 수 없다. 그룹 분할은 '겹치지 않는' 쪽의 극단, 공식 분할은 '형제가 적은' 쪽의 극단으로 본다.

## 5. 샘플러 점검 (그룹 분할 학습 ID, 에포크 0, 실제 배치 기준)

- 배치 크기 16, B의 배치당 비교 쌍 최대 4개. 학습 분할 안 명확한 쌍 7345개 중 이번 에포크 선택 800개
- A/B 같은 ID 집합·노출 횟수: True, 같은 배치 크기 구성: True, 배치에 들어간 검증 ID: 0

| 배치 | 배치 수 | 같은 배치의 형제 쌍 | 그중 명확한 쌍 | 배치당 명확한 쌍 | 조건 종류 |
|---|---|---|---|---|---|
| A | 200 | 46 | 37 | 0.185 | {'color': 16, 'relation_direction': 13, 'relation_kind': 9, 'pattern': 5, 'count': 4, 'pose': 4, 'material': 3, 'shape': 1} |
| B | 200 | 854 | 841 | 4.205 | {'color': 506, 'relation_direction': 344, 'relation_kind': 150, 'pattern': 93, 'count': 90, 'pose': 79, 'material': 43, 'state': 15, 'shape': 12} |

## 6. 대표 쌍 (`examples.jpg`)

| # | 조건 | A | B | 그룹 | 명시적 차이 |
|---|---|---|---|---|---|
| 1 | color | 2729 | 3514 | G0562 | 색[pumpkin] orange→green |
| 2 | color | 3170 | 3860 | G0190 | 색[dumbbell] red→black |
| 3 | color | 1730 | 2248 | G0465 | 색[cone] pink→green |
| 4 | relation_direction | 1685 | 2785 | G0612 | 관계 방향[block–kick scooter] block left_of kick scooter→kick scooter left_of block |
| 5 | relation_direction | 0314 | 3326 | G0256 | 관계 방향[horse–truck] truck left_of horse→horse left_of truck |
| 6 | relation_direction | 1795 | 3765 | G0026 | 관계 방향[baseball bat–camera] baseball bat left_of camera→camera left_of baseball bat |
| 7 | relation_kind | 0234 | 3512 | G0198 | 관계 종류[eraser–flower pot] eraser inside flower pot→eraser next_to flower pot |
| 8 | relation_kind | 2547 | 3627 | G0445 | 관계 종류[cylinder–food container] cylinder inside food container→cylinder next_to food container |
| 9 | relation_kind | 1572 | 2069 | G0422 | 관계 종류[bench–jacket] jacket on bench→bench next_to jacket |
| 10 | pattern | 1738 | 2558 | G0317 | 무늬[sofa] solid-colored→striped |
| 11 | pattern | 1272 | 1799 | G0564 | 무늬[sneaker] striped→solid-colored |
| 12 | pattern | 2410 | 3125 | G0487 | 무늬[bowl] solid-colored→striped |
| 13 | material | 1803 | 2754 | G0578 | 재질[suitcase] plastic→metal |
| 14 | material | 0269 | 3539 | G0224 | 재질[desk] wooden→metal |
| 15 | material | 1931 | 3530 | G0319 | 재질[fork] metal→wooden |
| 16 | pose | 1447 | 3107 | G0158 | 자세[donkey] standing→lying |
| 17 | pose | 2511 | 3241 | G0333 | 자세[person] sitting→lying |
| 18 | pose | 0888 | 2090 | G0116 | 자세[person] standing→kneeling |
| 19 | state | 0731 | 1980 | G0438 | 상태[cabinet] open→closed |
| 20 | state | 0091 | 0096 | G0083 | 상태[cabinet] closed→open |
| 21 | state | 2142 | 3340 | G0156 | 상태[backpack] closed→open |
| 22 | shape | 0915 | 1576 | G0036 | 모양[bottle] round→square |
| 23 | shape | 0748 | 1710 | G0376 | 모양[pitcher] round→square |
| 24 | shape | 0760 | 3097 | G0036 | 모양[bottle] round→square |
| 25 | none | 0773 | 2011 | G0452 |  |
| 26 | none | 2112 | 2908 | G0606 |  |
| 27 | none | 1324 | 2443 | G0168 |  |

육안 확인 결과는 `manual_checks.csv`에 적고, 실행할 때 `pairs.csv`·`probe_set.csv`의 `image_check(사람)` 열로 합친다.

## 7. 단일 속성 쌍, 점검 세트, 대조 문구 (후속 재정렬 실험용 전처리)

- 범위: 그룹 분할 학습 ID 안의 쌍만 (최종 검증 세트는 쓰지 않음)
- 단일 속성 판정(전체 쌍): {'해당 없음': 8407, '단일(미확인)': 1671, '단일 확정': 2051}
  - `단일 확정`: 명시적 차이가 물체 조건 1개뿐이고, 한쪽에만 적힌 조건·동시에 참일 수 있는 관계·파싱 애매가 없음
  - `단일(미확인)`: 명시적 차이는 1개지만 다른 조건이 한쪽에서 생략되어 나머지가 같은지 확인할 수 없음
- 학습 분할 안 `단일 확정` 쌍 (조건별): {'relation_direction': 585, 'color': 450, 'relation_kind': 180, 'pattern': 159, 'pose': 105, 'material': 61, 'state': 24, 'shape': 18}
- 결합 교차 쌍 (두 물체의 색 또는 개수가 맞바뀜, 실제 데이터에서 발견. 한 쌍이 여러 번 셀 수 있음): 전체 {'색': 3263, '개수': 1196}, 학습 분할 안 {'색': 2577, '개수': 978}
- 대조 문구 행 (`contrast_phrases.csv`): {'color_binding_swap': 1987, 'count_binding_swap': 941, 'color': 450, 'relation_left_right': 323, 'relation_above_below': 255, 'pattern': 159, 'pose': 105, 'material': 61, 'state': 24, 'shape': 18}
- 점검 세트 (`probe_set.csv`, `probe_set.jpg`): {'color': 6, 'count_binding_swap': 6, 'state': 6, 'relation_left_right': 6, 'color_binding_swap': 6}
- 대조 문구는 **판별력 점검 전용**이다(`use = probe_only`). 정답 캡션이나 학습 데이터로 저장·사용하지 않는다. 두 실제 설명의 값으로만 만들고, 한 물체에 값이 여러 개이거나 같은 종류 물체가 여럿이거나 관계가 left/above 외인 경우는 건너뛴다.
- 이후 사용 규칙: 판별 비율은 학습 분할 안 개발용 쌍으로만 잰다. 최종 검증 세트는 조건 선택 기준을 정하는 데 쓰지 않는다. 검색 시 각 질의의 순위는 그 질의와 후보 이미지로만 계산하고, 후보 이미지의 짝 설명은 쓰지 않는다.
