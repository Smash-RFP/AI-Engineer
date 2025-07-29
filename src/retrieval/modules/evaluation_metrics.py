import numpy as np
import time
from typing import List, Dict

from config import EVAL_PARAMS
from config import TOP_K


def precision_at_k(results, ground_truths, k):
    precision_scores = []
    for i in range(len(results)):
        retrieved = set(results[i]["retrieved_ids"][:k])
        relevant = set(ground_truths[i])
        if not retrieved:
            precision_scores.append(0.0)
        else:
            precision_scores.append(len(retrieved & relevant) / len(retrieved))
    return round(sum(precision_scores) / len(precision_scores), 4)


def recall_at_k(results, ground_truths, k):
    recall_scores = []
    for i in range(len(results)):
        retrieved = set(results[i]["retrieved_ids"][:k])
        relevant = set(ground_truths[i])
        if not relevant:
            recall_scores.append(0.0)
        else:
            recall_scores.append(len(retrieved & relevant) / len(relevant))
    return round(sum(recall_scores) / len(recall_scores), 4)


def mean_reciprocal_rank(results, ground_truths):
    rr_total = 0.0
    for i in range(len(results)):
        retrieved = results[i]["retrieved_ids"]
        relevant_set = set(ground_truths[i])
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        rr_total += rr
    return round(rr_total / len(results), 4)


def f1_at_k(results, ground_truths, k):
    f1_scores = []
    for i in range(len(results)):
        retrieved = set(results[i]["retrieved_ids"][:k])
        relevant = set(ground_truths[i])
        if not retrieved or not relevant:
            f1_scores.append(0.0)
        else:
            precision = len(retrieved & relevant) / len(retrieved)
            recall = len(retrieved & relevant) / len(relevant)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    return round(sum(f1_scores) / len(f1_scores), 4)


# 지연 시간 측정
# Query Latency
def query_latency(func, *args, **kwargs):
    start = time.time()
    _ = func(*args, **kwargs)
    end = time.time()
    return end - start



# Retrieval + 평가 통합 실행 코드
def normalize_id(text: str):
    import re
    text = text.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def evaluate_retrieval(results, ground_truths, k_values):
    scores = {
        'P@K': {},
        'R@K': {},
        'F1@K': {},
    }
    mrr = mean_reciprocal_rank(results, ground_truths)

    for k in k_values:
        scores['P@K'][k] = precision_at_k(results, ground_truths, k)
        scores['R@K'][k] = recall_at_k(results, ground_truths, k)
        scores['F1@K'][k] = f1_at_k(results, ground_truths, k)

    # best K 기준으로 대표값 정리 (예: 가장 높은 F1 기준)
    best_k = max(scores['F1@K'], key=scores['F1@K'].get)
    return {
        'P@K': round(scores['P@K'][best_k], 4),
        'R@K': round(scores['R@K'][best_k], 4),
        'F1@K': round(scores['F1@K'][best_k], 4),
        'MRR': round(mrr, 4),
        'best_k': best_k
    }
