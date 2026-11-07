"""Deterministic evaluation; no judge-model calls or implied faithfulness claims."""
import math
from collections import Counter
from app.application.search_pipeline import tokens


def retrieval_metrics(retrieved, relevant, k=5):
    if k < 1:
        raise ValueError('k must be positive')
    relevant = set(relevant)
    # Duplicates cannot inflate scores.
    ranked = list(dict.fromkeys(retrieved))[:k]
    hits = [int(item in relevant) for item in ranked]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {'precision_at_k': sum(hits) / k,
            'recall_at_k': sum(hits) / len(relevant) if relevant else 0.0,
            'mrr': next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
            'ndcg_at_k': dcg / ideal if ideal else 0.0}


def answer_metrics(answer, expected):
    left, right = Counter(tokens(answer)), Counter(tokens(expected))
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values()) if left else 0.0
    recall = overlap / sum(right.values()) if right else 0.0
    return {'exact_match': float(tokens(answer) == tokens(expected)),
            'token_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
