""" 전체 실험에 필요한 하이퍼파라미터, 경로, 시각화 설정을 한 곳에 모음 """

TOP_K = 5
HYBRID_ALPHA = 0.6    # 0.0 = pure BM25, 1.0 = pure Dense
COLLECTION_NAME = "rfp_documents"

# 경로
import os
# ai_engineer 기준 루트 경로 고정
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
BM25_DOCS_PATH = os.path.join(DATA_DIR, "bm25_docs.pkl")
