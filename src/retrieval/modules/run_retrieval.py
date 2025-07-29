import os
import time
import json
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


def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def run():
    # 설정
    persist_dir = VECTOR_DB_PATH
    bm25_docs_path = BM25_DOCS_PATH
    queries_path = "../data/queries.json"
    gt_path = "../data/ground_truth.json"
    result_dir = "../results"
    os.makedirs(result_dir, exist_ok=True)

    # 데이터 로딩
    print("⚡ 벡터스토어 로딩 중...")
    vectordb = load_vectorstore(persist_dir)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": max(EVAL_PARAMS['k_values'])})

    print("⚡ BM25 문서 로딩 및 초기화 중...")
    bm25_docs = load_bm25_documents(bm25_docs_path)
    bm25_retriever = init_bm25_retriever(bm25_docs)

    print("⚡ 쿼리 및 정답셋 로딩 중...")
    queries = load_queries(queries_path)
    ground_truths = load_ground_truth(gt_path)

    # 1. BM25-only Retrieval 실행
    print("===========================")
    print("⚡ BM25-only Retrieval 실행 중...")
    start_time = time.time()
    bm25_results = retrieve_documents(bm25_retriever, queries)
    bm25_elapsed = round(time.time() - start_time, 2)

    save_results(bm25_results, os.path.join(result_dir, "bm25_results.json"))
    bm25_metrics = evaluate_retrieval(bm25_results, ground_truths, k_values=EVAL_PARAMS['k_values'])
    print("BM25-only 성능:", bm25_metrics)

    # 2. Dense-only Retrieval 실행
    print("===========================")
    print("⚡ Dense-only Retrieval 실행 중...")
    start_time = time.time()
    dense_results = retrieve_documents(dense_retriever, queries)
    dense_elapsed = round(time.time() - start_time, 2)

    save_results(dense_results, os.path.join(result_dir, "dense_results.json"))
    dense_metrics = evaluate_retrieval(dense_results, ground_truths, k_values=EVAL_PARAMS['k_values'])
    print("Dense-only 성능:", dense_metrics)

    # 3. Hybrid Retrieval 실행
    print("===========================")
    print("⚡ Hybrid Retrieval 실행 중...")
    start_time = time.time()
    hybrid_results = hybrid_retrieve(bm25_retriever, dense_retriever, queries)
    hybrid_elapsed = round(time.time() - start_time, 2)

    save_results(hybrid_results, os.path.join(result_dir, "hybrid_results.json"))
    hybrid_metrics = evaluate_retrieval(hybrid_results, ground_truths, k_values=EVAL_PARAMS['k_values'])
    print("Hybrid 성능:", hybrid_metrics)

    print("===========================")
    print("\n전체 평가 완료!")
    print("- BM25 :", bm25_metrics)
    print("- Dense:", dense_metrics)
    print("- Hybrid:", hybrid_metrics)

    # k별 결과 기록
    for k in EVAL_PARAMS['k_values']:
        print(f"[LOG] BM25 (k={k}) 로그 저장 중...")
        log_experiment(
            experiment_name=f"BM25Test@{k}",
            mode="BM25",
            top_k=k,
            metrics={
                "P@K": bm25_metrics.get(f"P@{k}", 0),
                "R@K": bm25_metrics.get(f"R@{k}", 0),
                "F1@K": bm25_metrics.get(f"F1@{k}", 0),
                "MRR": bm25_metrics.get(f"MRR@{k}", 0)
            },
            elapsed_time=bm25_elapsed,
            avg_latency=compute_avg_latency(bm25_results),
            notes=f"BM25-only 실험 (k={k})"
        )

        print(f"[LOG] Dense (k={k}) 로그 저장 중...")
        log_experiment(
            experiment_name=f"DenseTest@{k}",
            mode="Dense",
            top_k=k,
            metrics={
                "P@K": dense_metrics.get(f"P@{k}", 0),
                "R@K": dense_metrics.get(f"R@{k}", 0),
                "F1@K": dense_metrics.get(f"F1@{k}", 0),
                "MRR": dense_metrics.get(f"MRR@{k}", 0)
            },
            elapsed_time=dense_elapsed,
            avg_latency=compute_avg_latency(dense_results),
            dense_model="text-embedding-3-small",
            notes=f"Dense-only 실험 (k={k})"
        )

        print(f"[LOG] Hybrid (k={k}) 로그 저장 중...")
        log_experiment(
            experiment_name=f"HybridTest@{k}",
            mode="Hybrid",
            top_k=k,
            metrics={
                "P@K": hybrid_metrics.get(f"P@{k}", 0),
                "R@K": hybrid_metrics.get(f"R@{k}", 0),
                "F1@K": hybrid_metrics.get(f"F1@{k}", 0),
                "MRR": hybrid_metrics.get(f"MRR@{k}", 0)
            },
            elapsed_time=hybrid_elapsed,
            avg_latency=compute_avg_latency(hybrid_results),
            hybrid_alpha=HYBRID_ALPHA,
            notes=f"Hybrid 실험 (k={k})"
        )

if __name__ == "__main__":
    run()