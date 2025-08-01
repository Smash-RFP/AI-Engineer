import numpy as np
import time
from typing import List, Dict, Union
from collections import defaultdict
from config import EVAL_PARAMS
from config import TOP_K

# 공통 ID 생성 함수 (source_id + chunk_id)
def make_chunk_identifier(source_id: str, chunk_id: str) -> str:
    # chunk_id가 이미 'chunk-'로 시작한다면 중복 방지
    if chunk_id.startswith("chunk-"):
        return f"{source_id}_chunk_{chunk_id}"
    return f"{source_id}_chunk_chunk-{chunk_id}"


# Precision@K (Document 기준)
def doc_level_precision_at_k(results: List[Dict], ground_truths: List[Dict], k: int) -> float:
    precisions = []
    for result, gt in zip(results, ground_truths):
        retrieved_docs = [r["retrieved_source_id"] for r in result["results"][:k]]
        gt_docs = set(g["source_id"] for g in gt["ground_truths"])
        if not retrieved_docs:
            precisions.append(0.0)
        else:
            precisions.append(len(set(retrieved_docs) & gt_docs) / len(retrieved_docs))
    return round(np.mean(precisions), 4)


# Recall@K (Document 기준)
def doc_level_recall_at_k(results: List[Dict], ground_truths: List[Dict], k: int) -> float:
    recalls = []
    for result, gt in zip(results, ground_truths):
        retrieved_docs = [r["retrieved_source_id"] for r in result["results"][:k]]
        gt_docs = set(g["source_id"] for g in gt["ground_truths"])
        if not gt_docs:
            recalls.append(0.0)
        else:
            recalls.append(len(set(retrieved_docs) & gt_docs) / len(gt_docs))
    return round(np.mean(recalls), 4)


# Precision@K (Chunk 기준)
def chunk_level_precision_at_k(results: List[Dict], ground_truths: List[Dict], k: int) -> float:
    precisions = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = [make_chunk_identifier(r["retrieved_source_id"], r["retrieved_chunk_id"]) for r in result["results"][:k]]
        gt_chunks = set(make_chunk_identifier(g["source_id"], g["chunk_id"]) for g in gt["ground_truths"])
        if not retrieved_chunks:
            precisions.append(0.0)
        else:
            precisions.append(len(set(retrieved_chunks) & gt_chunks) / len(retrieved_chunks))
    return round(np.mean(precisions), 4)


# Recall@K (Chunk 기준)
def chunk_level_recall_at_k(results: List[Dict], ground_truths: List[Dict], k: int) -> float:
    recalls = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = [make_chunk_identifier(r["retrieved_source_id"], r["retrieved_chunk_id"]) for r in result["results"][:k]]
        gt_chunks = set(make_chunk_identifier(g["source_id"], g["chunk_id"]) for g in gt["ground_truths"])
        if not gt_chunks:
            recalls.append(0.0)
        else:
            recalls.append(len(set(retrieved_chunks) & gt_chunks) / len(gt_chunks))
    return round(np.mean(recalls), 4)


# 전체 평가
def evaluate_2stage_retrieval(results: List[Dict], ground_truths: List[Dict], k_values: List[int]) -> Dict[str, Dict[int, float]]:
    scores = defaultdict(dict)
    for k in k_values:
        scores["Doc_P@K"][k] = doc_level_precision_at_k(results, ground_truths, k)
        scores["Doc_R@K"][k] = doc_level_recall_at_k(results, ground_truths, k)
        scores["Chunk_P@K"][k] = chunk_level_precision_at_k(results, ground_truths, k)
        scores["Chunk_R@K"][k] = chunk_level_recall_at_k(results, ground_truths, k)
    return scores

