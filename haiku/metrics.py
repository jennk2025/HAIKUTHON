"""공식 평가식: query별 0.5 × (Recall@1 + NDCG@5)의 평균."""

import math

import numpy as np


def query_score(rank):
    """rank: 정답 순위(1부터). None이면 5위 밖."""
    if rank is None or rank > 5:
        return 0.0
    return 0.5 * (float(rank == 1) + 1.0 / math.log2(rank + 1))


def metrics_from_ranks(ranks, hit_ks=(5, 10, 20, 50, 100)):
    """전체 순위 배열로 공식 점수와 Hit@K, 순위 분포를 계산한다."""
    r = np.asarray(ranks, dtype=np.int64)
    recall1 = float((r == 1).mean())
    ndcg5 = float(np.where(r <= 5, 1.0 / np.log2(r + 1), 0.0).mean())
    out = {
        "n_queries": int(len(r)),
        "score": 0.5 * (recall1 + ndcg5),
        "recall@1": recall1,
        "ndcg@5": ndcg5,
    }
    for k in hit_ks:
        out[f"hit@{k}"] = float((r <= k).mean())
    out["rank_dist"] = {
        "1": int((r == 1).sum()),
        "2-5": int(((r >= 2) & (r <= 5)).sum()),
        "6-20": int(((r >= 6) & (r <= 20)).sum()),
        ">20": int((r > 20).sum()),
    }
    return out


def score_predictions(predictions, gold, top_k=5):
    """predict()의 반환값 {prompt_id: [image_id × 5]}를 검사하고 공식 점수를 계산한다."""
    missing = [q for q in gold if q not in predictions]
    if missing:
        raise ValueError(f"예측이 없는 query {len(missing)}개 (예: {missing[:3]})")
    recall1 = ndcg5 = 0.0
    for qid, target in gold.items():
        ranking = list(predictions[qid])
        if len(ranking) != top_k or len(set(ranking)) != top_k:
            raise ValueError(f"{qid}: Top-{top_k}가 아니거나 중복이 있습니다: {ranking}")
        if target in ranking:
            rank = ranking.index(target) + 1
            recall1 += float(rank == 1)
            ndcg5 += 1.0 / math.log2(rank + 1)
    n = len(gold)
    return {"n_queries": n, "score": 0.5 * (recall1 + ndcg5) / n, "recall@1": recall1 / n, "ndcg@5": ndcg5 / n}
