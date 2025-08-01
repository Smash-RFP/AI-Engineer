import os
import json
import time
from typing import List, Dict
import pickle

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import TOP_K, HYBRID_ALPHA, COLLECTION_NAME
from config import VECTOR_DB_PATH

def load_vectorstore(persist_dir=VECTOR_DB_PATH) -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )

def load_bm25_documents(path: str) -> List[Document]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def init_bm25_retriever(docs: List[Document], k: int = TOP_K) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

def min_max_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    values = list(score_dict.values())
    min_s, max_s = min(values), max(values)
    if max_s - min_s == 0:
        return {k: 0.0 for k in score_dict}
    return {k: (v - min_s) / (max_s - min_s) for k, v in score_dict.items()}

def z_score_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    values = list(score_dict.values())
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return {k: 0.0 for k in score_dict}
    return {k: (v - mean) / std for k, v in score_dict.items()}

def extract_doc_info(doc: Document) -> Dict:
    source_id = doc.metadata.get("source_id", "")
    chunk_id = doc.metadata.get("chunk_id", "unknown")
    return {
        "retrieved_source_id": source_id,
        "retrieved_chunk_id": chunk_id,
        "retrieved_content": doc.page_content[:500]
    }

def hybrid_retrieve(bm25_retriever, dense_retriever, queries: List[str], alpha: float = HYBRID_ALPHA) -> List[Dict]:
    results = []
    for query in queries:
        start = time.perf_counter()
        bm25_docs = bm25_retriever.invoke(query)
        dense_docs = dense_retriever.invoke(query)

        doc_scores = {}
        all_docs = {}

        for idx, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("source_id", "")
            doc_scores.setdefault(doc_id, {})["bm25"] = 1 / (idx + 1)
            all_docs[doc_id] = doc

        for idx, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("source_id", "")
            doc_scores.setdefault(doc_id, {})["dense"] = 1 / (idx + 1)
            all_docs[doc_id] = doc

        bm25_norm = z_score_normalize({k: v.get("bm25", 0) for k, v in doc_scores.items()})
        dense_norm = z_score_normalize({k: v.get("dense", 0) for k, v in doc_scores.items()})

        combined = {
            doc_id: alpha * dense_norm.get(doc_id, 0) + (1 - alpha) * bm25_norm.get(doc_id, 0)
            for doc_id in doc_scores
        }

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        end = time.perf_counter()

        entries = [extract_doc_info(all_docs[doc_id]) for doc_id, _ in sorted_docs[:TOP_K]]

        results.append({
            "query": query,
            "results": entries,
            "latency_sec": round(end - start, 4)
        })
    return results

def re_rank(results: List[Dict], model: CrossEncoder, top_k: int = TOP_K) -> List[Dict]:
    reranked_results = []

    for r in results:
        query = r["query"]
        retrieved_ids = r.get("retrieved_ids", [])
        contents = r.get("retrieved_contents", [])
        start = time.perf_counter()

        if not contents or not retrieved_ids:
            reranked_results.append({
                "query": query,
                "results": [],
                "latency_sec": r.get("latency_sec", 0.0)
            })
            continue

        # 쿼리-패시지 쌍 스코어링
        scores = model.predict([(query, passage) for passage in contents])
        ranked = sorted(zip(retrieved_ids, contents, scores), key=lambda x: x[2], reverse=True)

        entries = []
        for doc_id, content, _ in ranked[:top_k]:
            entries.append({
                "retrieved_source_id": doc_id.get("source_id", ""),
                "retrieved_chunk_id": doc_id.get("chunk_id", "unknown"),
                "retrieved_content": content
            })

        end = time.perf_counter()

        reranked_results.append({
            "query": query,
            "results": entries,
            "latency_sec": round(end - start, 4)
        })
    return reranked_results


def retrieve_documents(retriever, queries: List[str], use_rerank=False, model: CrossEncoder = None) -> List[Dict]:
    results = []
    for query in queries:
        start = time.perf_counter()
        docs = retriever.invoke(query)
        end = time.perf_counter()

        if use_rerank and model:
            ids = [doc.metadata.get("source_id", "") for doc in docs]
            contents = [doc.page_content[:500] for doc in docs]
            reranked = re_rank([{
                "query": query,
                "retrieved_ids": ids,
                "retrieved_contents": contents,
                "latency_sec": round(end - start, 4)
            }], model=model)
            results.extend(reranked)
        else:
            entries = [extract_doc_info(doc) for doc in docs[:TOP_K]]
            results.append({
                "query": query,
                "results": entries,
                "latency_sec": round(end - start, 4)
            })
    return results

def save_results(results: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def compute_avg_latency(results: List[Dict]) -> float:
    latencies = [r["latency_sec"] for r in results if "latency_sec" in r]
    return round(sum(latencies) / len(latencies), 4) if latencies else 0.0
