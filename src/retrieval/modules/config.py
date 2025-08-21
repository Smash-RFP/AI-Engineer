# retrieve

# 경로
import os
# ai_engineer 기준 루트 경로 고정
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
CLEANED_CHUNK_DIR = os.path.join(DATA_DIR, "cleaned_chunks")
BM25_DOCS_PATH = os.path.join(DATA_DIR, "bm25_docs.pkl")
BM25_MAP_PATH = os.path.join(DATA_DIR, "bm25_chunk_id_map.json")
META_EMBEDDING_PATH = os.path.join(DATA_DIR, "meta_embedding_dict.pkl")

# 하이퍼파라미터
HYBRID_ALPHA = 0.4  # dense
HYBRID_BETA = 0.3   # bm25
HYBRID_GAMMA = 0.3  # meta

TOP_K = 5
COLLECTION_NAME = "rfp_documents"

STRATEGY = "hybrid" # or "dense", "bm25"
NORMALIZATION_METHOD = "z_score"  # or "softmax", "min_max"



