import os
import time
import json
from sentence_transformers import CrossEncoder

from retrieval import (
    hybrid_retrieve,
    retrieve_documents,
    load_vectorstore,
    init_bm25_retriever,
    load_bm25_documents
)
from evaluation_metrics import evaluate_retrieval
from retrieval import load_queries, load_ground_truth, compute_avg_latency
from config import TOP_K, HYBRID_ALPHA, COLLECTION_NAME, EVAL_PARAMS
from config import BM25_DOCS_PATH, VECTOR_DB_PATH
from experiment_logger import log_experiment, compute_avg_latency

from retrieval import re_rank


def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def run():
    persist_dir = VECTOR_DB_PATH
    bm25_docs_path = BM25_DOCS_PATH
    queries_path = "../data/queries.json"
    gt_path = "../data/ground_truth.json"
    result_dir = "../results"
    os.makedirs(result_dir, exist_ok=True)

    print("⚡ 벡터스토어 로딩 중...")
    vectordb = load_vectorstore(persist_dir)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": max(EVAL_PARAMS['k_values'])})

    print("⚡ BM25 문서 로딩 및 초기화 중...")
    bm25_docs = load_bm25_documents(bm25_docs_path)
    bm25_retriever = init_bm25_retriever(bm25_docs)

    print("⚡ 쿼리 및 정답셋 로딩 중...")
    queries = load_queries(queries_path)
    ground_truths = load_ground_truth(gt_path)

    # --- BM25 Retrieval ---
    print("\n⚡ BM25-only Retrieval 실행 중...")
    start = time.time()
    bm25_results = retrieve_documents(bm25_retriever, queries)
    bm25_elapsed = time.time() - start
    save_results(bm25_results, os.path.join(result_dir, "bm25_results.json"))
    bm25_metrics = evaluate_retrieval(bm25_results, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("BM25-only 성능:", bm25_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="BM25Test",
            mode="BM25",
            # top_k=bm25_metrics["best_k"],
            k=k,
            metrics=bm25_metrics,
            elapsed_time=bm25_elapsed,
            avg_latency=compute_avg_latency(bm25_results),
            notes=f"BM25-only 실험 (k={k})"
        )

    # --- Dense Retrieval ---
    print("\n⚡ Dense-only Retrieval 실행 중...")
    start = time.time()
    dense_results = retrieve_documents(dense_retriever, queries)
    dense_elapsed = time.time() - start
    save_results(dense_results, os.path.join(result_dir, "dense_results.json"))
    dense_metrics = evaluate_retrieval(dense_results, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("Dense-only 성능:", dense_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="DenseTest",
            mode="Dense",
            # top_k=dense_metrics["best_k"],
            k=k,
            metrics=dense_metrics,
            elapsed_time=dense_elapsed,
            avg_latency=compute_avg_latency(dense_results),
            dense_model="text-embedding-3-small",
            notes=f"Dense-only 실험 (k={k})"
        )

    # --- Hybrid Retrieval ---
    print("\n⚡ Hybrid Retrieval 실행 중...")
    start = time.time()
    hybrid_results = hybrid_retrieve(bm25_retriever, dense_retriever, queries)
    hybrid_elapsed = time.time() - start
    save_results(hybrid_results, os.path.join(result_dir, "hybrid_results.json"))
    hybrid_metrics = evaluate_retrieval(hybrid_results, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("Hybrid 성능:", hybrid_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="HybridTest",
            mode="Hybrid",
            # top_k=hybrid_metrics["best_k"],
            k=k,
            metrics=hybrid_metrics,
            elapsed_time=hybrid_elapsed,
            avg_latency=compute_avg_latency(hybrid_results),
            hybrid_alpha=HYBRID_ALPHA,
            notes=f"Hybrid 실험 (k={k})"
        )
        
    
    # --- Cross-Encoder 모델 로딩 ---
    print("⚡ Cross-Encoder 모델 로딩 중...")
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- BM25 + Cross-Encoder ---
    print("\n⚡ BM25 + CrossEncoder Re-ranking 중...")
    bm25_reranked = re_rank(bm25_results, model=cross_encoder_model)
    save_results(bm25_reranked, os.path.join(result_dir, "bm25_reranked_results.json"))
    bm25_reranked_metrics = evaluate_retrieval(bm25_reranked, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("BM25 + CrossEncoder 성능:", bm25_reranked_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="BM25_Reranked",
            mode="BM25+CrossEncoder",
            k=k,
            metrics=bm25_reranked_metrics,
            elapsed_time=bm25_elapsed,  # 재정렬 포함 시간 측정 필요 시 갱신
            avg_latency=compute_avg_latency(bm25_reranked),
            notes=f"BM25 + CrossEncoder 재정렬 (k={k})"
        )

    # --- Dense + Cross-Encoder ---
    print("\n⚡ Dense + CrossEncoder Re-ranking 중...")
    dense_reranked = re_rank(dense_results, model=cross_encoder_model)
    save_results(dense_reranked, os.path.join(result_dir, "dense_reranked_results.json"))
    dense_reranked_metrics = evaluate_retrieval(dense_reranked, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("Dense + CrossEncoder 성능:", dense_reranked_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="Dense_Reranked",
            mode="Dense+CrossEncoder",
            k=k,
            metrics=dense_reranked_metrics,
            elapsed_time=dense_elapsed,
            avg_latency=compute_avg_latency(dense_reranked),
            dense_model="text-embedding-3-small",
            notes=f"Dense + CrossEncoder 재정렬 (k={k})"
        )

    # --- Hybrid + Cross-Encoder ---
    print("\n⚡ Hybrid + CrossEncoder Re-ranking 중...")
    hybrid_reranked = re_rank(hybrid_results, model=cross_encoder_model)
    save_results(hybrid_reranked, os.path.join(result_dir, "hybrid_reranked_results.json"))
    hybrid_reranked_metrics = evaluate_retrieval(hybrid_reranked, ground_truths, k_values=EVAL_PARAMS['k_values'])

    print("Hybrid + CrossEncoder 성능:", hybrid_reranked_metrics)

    for k in EVAL_PARAMS['k_values']:
        log_experiment(
            experiment_name="Hybrid_Reranked",
            mode="Hybrid+CrossEncoder",
            k=k,
            metrics=hybrid_reranked_metrics,
            elapsed_time=hybrid_elapsed,
            avg_latency=compute_avg_latency(hybrid_reranked),
            hybrid_alpha=HYBRID_ALPHA,
            notes=f"Hybrid + CrossEncoder 재정렬 (k={k})"
        )
        
    


if __name__ == "__main__":
    run()