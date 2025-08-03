import os
import json
from sentence_transformers import CrossEncoder

from config import TOP_K, HYBRID_ALPHA, EVAL_PARAMS
from config import BM25_DOCS_PATH, VECTOR_DB_PATH
from retrieval import (
    hybrid_retrieve, retrieve_documents, load_vectorstore,
    init_bm25_retriever, load_bm25_documents, load_json, compute_avg_latency, re_rank, generate_hypothetical_passage
)
from evaluation_metrics import evaluate_2stage_retrieval
from experiment_logger import log_2stage_experiment

def check_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print("설정 완료")
    else:
        print("설정 안됨")

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_2stage_eval(results, ground_truths, label):
    two_stage_scores = evaluate_2stage_retrieval(results, ground_truths, k_values=EVAL_PARAMS['k_values'])
    print(f"\n{label} (문서/청크 기준 2단계 평가):")
    for k in EVAL_PARAMS["k_values"]:
        print(f"K={k} | Doc_P: {two_stage_scores['Doc_P@K'][k]} | Doc_R: {two_stage_scores['Doc_R@K'][k]} | "
              f"Chunk_P: {two_stage_scores['Chunk_P@K'][k]} | Chunk_R: {two_stage_scores['Chunk_R@K'][k]}")


def log_all_k(experiment_name, mode, results, scores, model_info=None):
    for k in EVAL_PARAMS["k_values"]:
        log_2stage_experiment(
            experiment_name=experiment_name,
            mode=mode,
            k=k,
            metrics=scores,
            elapsed_time=0.0,
            avg_latency=compute_avg_latency(results),
            hybrid_alpha=HYBRID_ALPHA if "Hybrid" in mode else None,
            dense_model=model_info if "Dense" in mode else None,
            notes=f"{mode} 실험 (k={k})"
        )


def run():
    # 경로 설정
    persist_dir = VECTOR_DB_PATH
    bm25_docs_path = BM25_DOCS_PATH
    queries_path = "src/retrieval/data/ground_truth.json"
    result_dir = "src/retrieval/results"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1) BM25용 매핑
    bm25_map_path = "src/retrieval/data/bm25_chunk_id_map.json"
    bm25_chunk_map = load_json(bm25_map_path)

    # 2) Chroma(Dense)용 매핑
    chroma_map_path = "src/vectordb/chunk_id_map.json"
    chroma_chunk_map = load_json(chroma_map_path)

    # 데이터 로딩
    print("⚡ 벡터스토어 로딩 중...")
    vectordb = load_vectorstore(persist_dir)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": max(EVAL_PARAMS['k_values'])})

    print("⚡ BM25 문서 로딩 및 초기화 중...")
    bm25_docs = load_bm25_documents(bm25_docs_path)
    bm25_retriever = init_bm25_retriever(bm25_docs)

    # print("⚡ 쿼리 및 정답셋 로딩 중...")
    # queries = load_json(queries_path)
    # ground_truths = load_json(gt_path)
    
    print("⚡ HyDE 기반 쿼리 생성 중...")
    # 기존 ground_truth.json을 로딩
    ground_truth_data = load_json(queries_path)

    # 평가용 ground_truths
    ground_truths = [item for item in ground_truth_data]
    
    # 쿼리만 추출해서 HyDE에 넣기
    queries = [item["query"] for item in ground_truth_data]
    
    # HyDE 생성 문서
    hyde_queries = [generate_hypothetical_passage(q) for q in queries]

    chunk_id_map_path = "/home/gcp-JeOn/Smash-RFP/src/vectordb/chunk_id_map.json"
    with open(chunk_id_map_path, "r", encoding="utf-8") as f:
        chunk_id_map = json.load(f)

    # 1단계 Retrieval
    retrieval_modes = [
        # HyDE+BM25
        ("BM25Test", "BM25",
         retrieve_documents(bm25_retriever, hyde_queries,
                            use_rerank=False,
                            model=None,
                            chunk_id_map=bm25_chunk_map),
         "HyDE"
        ),
        # HyDE+Dense
        ("DenseTest", "Dense",
         retrieve_documents(dense_retriever, hyde_queries,
                            use_rerank=False,
                            model=None,
                            chunk_id_map=chroma_chunk_map),
         "text-embedding-3-small, HyDE"
        ),
        # HyDE+Hybrid
        ("HybridTest", "Hybrid",
         hybrid_retrieve(bm25_retriever,
                         dense_retriever,
                         hyde_queries,
                         alpha=HYBRID_ALPHA,
                         bm25_chunk_id_map=bm25_chunk_map,
                         dense_chunk_id_map=chroma_chunk_map),
         "HyDE"
        ),
    ]

    for exp_name, mode, results, model_info in retrieval_modes:
        print(f"\n⚡ {mode} Retrieval 실행 중...")
        save_results(results, os.path.join(result_dir, f"{mode.lower()}_results.json"))
        scores = evaluate_2stage_retrieval(results, ground_truths, k_values=EVAL_PARAMS['k_values'])
        print(f"{mode} 성능:", scores)
        print_2stage_eval(results, ground_truths, mode)
        log_all_k(exp_name, mode, results, scores, model_info)

    # 2단계 CrossEncoder Re-ranking
    print("\n⚡ Cross-Encoder 모델 로딩 중...")
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    rerank_modes = [
    (
        "BM25_CrossEncoderTest", "BM25+CrossEncoder",
        re_rank([
            {
                "query": r["query"],
                "retrieved_ids": [
                    {
                        "source_id": e["retrieved_source_id"],
                        "chunk_id": e["retrieved_chunk_id"]
                    } for e in r["results"]
                ],
                "retrieved_contents": [e["retrieved_content"] for e in r["results"]],
                "latency_sec": r["latency_sec"]
            } for r in retrieval_modes[0][2]
        ], model=cross_encoder_model),
        "HyDE"
    ),
    (
        "Dense_CrossEncoderTest", "Dense+CrossEncoder",
        re_rank([
            {
                "query": r["query"],
                "retrieved_ids": [
                    {
                        "source_id": e["retrieved_source_id"],
                        "chunk_id": e["retrieved_chunk_id"]
                    } for e in r["results"]
                ],
                "retrieved_contents": [e["retrieved_content"] for e in r["results"]],
                "latency_sec": r["latency_sec"]
            } for r in retrieval_modes[1][2]
        ], model=cross_encoder_model),
        "text-embedding-3-small"
    ),
    (
        "Hybrid_CrossEncoderTest", "Hybrid+CrossEncoder",
        re_rank([
            {
                "query": r["query"],
                "retrieved_ids": [
                    {
                        "source_id": e["retrieved_source_id"],
                        "chunk_id": e["retrieved_chunk_id"]
                    } for e in r["results"]
                ],
                "retrieved_contents": [e["retrieved_content"] for e in r["results"]],
                "latency_sec": r["latency_sec"]
            } for r in retrieval_modes[2][2]
        ], model=cross_encoder_model),
        "HyDE"
    )
]

    for exp_name, mode, reranked_results, model_info in rerank_modes:
        print(f"\n⚡ {mode} Re-ranking 결과 저장 및 평가 중...")
        save_results(reranked_results, os.path.join(result_dir, f"{mode.lower()}_results.json"))
        rerank_scores = evaluate_2stage_retrieval(reranked_results, ground_truths, k_values=EVAL_PARAMS['k_values'])
        print(f"{mode} 성능:", rerank_scores)
        print_2stage_eval(reranked_results, ground_truths, mode)
        log_all_k(exp_name, mode, reranked_results, rerank_scores, model_info)

if __name__ == "__main__":
    check_api_keys()
    run()

