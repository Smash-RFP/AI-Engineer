import os
import json
import time
from typing import List, Dict
import pickle

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever

from config import TOP_K, HYBRID_ALPHA, COLLECTION_NAME
from config import VECTOR_DB_PATH
from evaluation_metrics import precision_at_k, recall_at_k, f1_at_k, mean_reciprocal_rank


# 벡터스토어 로딩
def load_vectorstore(persist_dir=VECTOR_DB_PATH) -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding, 
        collection_name=COLLECTION_NAME
        )
    return vectordb

# query 로딩
def load_queries(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
# 정답 로딩
def load_ground_truth(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    

# BM25 Retriever 초기화 코드
def init_bm25_retriever(docs: List[Document], k: int = TOP_K) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

def load_bm25_documents(path: str) -> List[Document]:
    with open(path, "rb") as f:
        docs = pickle.load(f)
    return docs


# Hybrid Retrieval (BM25 + Dense)
def hybrid_retrieve(bm25_retriever, dense_retriever, queries: List[str], alpha: float = HYBRID_ALPHA) -> List[Dict]:
    results = []
    for query in queries:
        start = time.perf_counter()

        bm25_docs = bm25_retriever.get_relevant_documents(query)
        dense_docs = dense_retriever.get_relevant_documents(query)

        combined_scores = {}
        doc_contents = {}  # content 저장용
        # BM25
        for idx, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("source_id", "")
            score = (1 / (idx + 1)) * (1 - alpha)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score
            doc_contents[doc_id] = doc.page_content[:300]

        # Dense
        for idx, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("source_id", "")
            score = (1 / (idx + 1)) * alpha
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score
            doc_contents[doc_id] = doc.page_content[:300]

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_ids = [doc_id for doc_id, _ in sorted_docs[:TOP_K]]
        retrieved_contents = [doc_contents[doc_id] for doc_id in retrieved_ids]

        end = time.perf_counter()

        results.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "retrieved_contents": retrieved_contents,
            "latency_sec": round(end - start, 4)
        })

    return results

# 단일 Retriever (BM25 or Dense) 실행
def retrieve_documents(retriever, queries: List[str]) -> List[Dict]:
    results = []
    for query in queries:
        start = time.perf_counter()

        docs = retriever.get_relevant_documents(query)
        retrieved_ids = [doc.metadata.get("source_id", "") for doc in docs]
        retrieved_contents = [doc.page_content[:300] for doc in docs]

        end = time.perf_counter()

        results.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "retrieved_contents": retrieved_contents,
            "latency_sec": round(end - start, 4)
        })
    return results


def save_results(results: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
        
def compute_avg_latency(results: List[Dict]) -> float:
    latencies = [r["latency_sec"] for r in results if "latency_sec" in r]
    return round(sum(latencies) / len(latencies), 4) if latencies else 0.0
