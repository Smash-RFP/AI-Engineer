""" 전체 실험에 필요한 하이퍼파라미터, 경로, 시각화 설정을 한 곳에 모음 """

# 평가지표 파라미터
EVAL_PARAMS = {
    'k_values': [1, 3, 5, 10],  # 여러 k 값으로 평가 가능
    'default_k': 10,            # 기본 k 값
    'latency_runs': 10,
}

# 시각화 파라미터
PLOT_CONFIG = {
    "colors": ["#4c72b0", "#55a868", "#c44e52"],
    "figsize": (10, 6),
    "title_fontsize": 14,
    "label_fontsize": 12,
}

TOP_K = 5
HYBRID_ALPHA = 0.5    # 0.0 = pure BM25, 1.0 = pure Dense
COLLECTION_NAME = "rfp_documents"

# 경로
VECTOR_DB_PATH = "src/vectordb/chroma_db"
BM25_DOCS_PATH = "src/retrieval/data/bm25_docs.pkl"