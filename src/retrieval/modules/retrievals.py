import os
import time
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from scipy.special import softmax

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    VECTOR_DB_PATH, BM25_DOCS_PATH, BM25_MAP_PATH,
    META_EMBEDDING_PATH, COLLECTION_NAME, TOP_K,
    HYBRID_ALPHA, HYBRID_BETA, HYBRID_GAMMA,
    NORMALIZATION_METHOD, STRATEGY
)

from .embedding_loader import load_embedding_model

# --- 유틸 함수 ---
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def try_int(value: str):
    try:
        return int(value)
    except:
        return value

def normalize_source_id(source_id: str) -> str:
    return source_id.strip().replace(" ", "_")

def normalize_scores(score_dict, method="z_score"):
    values = np.array(list(score_dict.values()))
    keys = list(score_dict.keys())
    if len(values) == 0: return {k: 0.0 for k in keys}
    if method == "z_score":
        mean, std = values.mean(), values.std()
        return {k: 0.0 if std == 0 else (v - mean) / std for k, v in score_dict.items()}
    elif method == "min_max":
        min_v, max_v = values.min(), values.max()
        return {k: 0.0 if max_v - min_v == 0 else (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}
    elif method == "softmax":
        sm = softmax(values)
        return {k: float(v) for k, v in zip(keys, sm)}
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# --- 검색 초기화 ---
def load_vectorstore(embedding_model):
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

def load_bm25_documents():
    return load_pickle(BM25_DOCS_PATH)

def init_bm25_retriever(docs): 
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = TOP_K
    return retriever


# --- 문서 정보 ---
def extract_doc_info(doc, chunk_id_map=None, scores=None):
    doc_id = doc.id
    meta = doc.metadata
    source_id = normalize_source_id(meta.get("source_id", "unknown"))
    chunk_id = chunk_id_map.get(doc_id, meta.get("chunk_id", "unknown")) if chunk_id_map else meta.get("chunk_id", "unknown")
    full_chunk_id = f"{source_id}_{try_int(chunk_id)}"
    info = {
        "retrieved_source_id": source_id,
        "retrieved_chunk_id": full_chunk_id,
        "retrieved_content": doc.page_content[:300]
    }
    if scores: info.update(scores)
    return info



# --- 단일 전략 검색 ---
def retrieve_documents(
    retriever, queries, use_rerank=False, rerank_model=None, chunk_id_map=None
):
    results = []
    for query in queries:
        start = time.perf_counter()
        docs = retriever.invoke(query)
        end = time.perf_counter()
        entries = []
        if use_rerank and rerank_model:
            contents = [doc.page_content[:512] for doc in docs]
            doc_ids = [extract_doc_info(doc, chunk_id_map) for doc in docs]
            scores = rerank_model.predict([(query, c) for c in contents])
            ranked = sorted(zip(doc_ids, contents, scores), key=lambda x: x[2], reverse=True)
            for doc_info, content, score in ranked[:TOP_K]:
                doc_info["retrieved_content"] = content
                doc_info["rerank_score"] = round(score, 4)
                entries.append(doc_info)
        else:
            for doc in docs[:TOP_K]:
                entry = extract_doc_info(doc, chunk_id_map)
                score_type = "dense_score" if isinstance(retriever, Chroma) else "bm25_score"
                entry[score_type] = 1.0
                entries.append(entry)

        results.append({"query": query, "results": entries, "latency_sec": round(end - start, 4)})
    return results


# --- Hybrid 검색 ---
def hybrid_retrieve(query, embedding_model, normalization_method=NORMALIZATION_METHOD):
    bm25_docs = load_bm25_documents()
    bm25_retriever = init_bm25_retriever(bm25_docs)
    dense_retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
    bm25_chunk_map = load_json(BM25_MAP_PATH)
    dense_chunk_map = load_json(BM25_MAP_PATH)
    meta_embedding_dict = load_pickle(META_EMBEDDING_PATH)

    query_embedding = embedding_model.embed_query(query)
    bm25_results = bm25_retriever.invoke(query)
    dense_results = dense_retriever.invoke(query)

    doc_scores = {}
    all_docs = {}

    # BM25
    for i, doc in enumerate(bm25_results):
        doc_scores.setdefault(doc.id, {})["bm25"] = 1 / (i + 1)
        all_docs[doc.id] = (doc, bm25_chunk_map)

    # Dense
    dense_embeddings = embedding_model.embed_documents([d.page_content for d in dense_results])
    for doc, emb in zip(dense_results, dense_embeddings):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
        doc_scores.setdefault(doc.id, {})["dense"] = sim
        all_docs[doc.id] = (doc, dense_chunk_map)

    # Meta
    for doc_id in doc_scores:
        doc, _ = all_docs[doc_id]
        sid = normalize_source_id(doc.metadata.get("source_id", ""))
        cid = try_int(doc.metadata.get("chunk_id", "unknown"))
        meta_key = f"{sid}_{cid}"
        meta_vec = meta_embedding_dict.get(meta_key)
        if meta_vec is not None:
            sim = np.dot(query_embedding, meta_vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(meta_vec) + 1e-8)
            doc_scores[doc_id]["meta"] = sim

    # 점수 결합
    bm25_norm = normalize_scores({k: v.get("bm25", 0.0) for k, v in doc_scores.items()}, normalization_method)
    dense_norm = normalize_scores({k: v.get("dense", 0.0) for k, v in doc_scores.items()}, normalization_method)
    meta_norm  = normalize_scores({k: v.get("meta",  0.0) for k, v in doc_scores.items()}, normalization_method)
    combined = {
        doc_id: HYBRID_ALPHA * dense_norm.get(doc_id, 0.0)
               + HYBRID_BETA * bm25_norm.get(doc_id, 0.0)
               + HYBRID_GAMMA * meta_norm.get(doc_id, 0.0)
        for doc_id in doc_scores
    }

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [
        extract_doc_info(all_docs[doc_id][0], all_docs[doc_id][1], {
            "bm25_score": bm25_norm.get(doc_id, 0.0),
            "dense_score": dense_norm.get(doc_id, 0.0),
            "meta_score": meta_norm.get(doc_id, 0.0),
            "hybrid_score": combined.get(doc_id, 0.0)
        }) for doc_id, _ in sorted_docs[:TOP_K]
    ]


# --- 최종 실행 함수 ---
def run_retrieve(query=None, strategy=STRATEGY, normalization_method=NORMALIZATION_METHOD, 
                 use_cross_encoder=True, embedding_model=None):
    if embedding_model is None:
        raise ValueError("❗ embedding_model must be provided to run_retrieve().")

    print(f"\nRetrieval 시작 → 전략: {strategy}, 정규화: {normalization_method}, rerank: {use_cross_encoder}")
    
    if strategy == "bm25":
        retriever = init_bm25_retriever(load_bm25_documents())
        results = retrieve_documents(retriever, [query], use_rerank=use_cross_encoder,
                                     rerank_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross_encoder else None,
                                     chunk_id_map=load_json(BM25_MAP_PATH))
        
    elif strategy == "dense":
        retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
        results = retrieve_documents(retriever, [query], use_rerank=use_cross_encoder,
                                     rerank_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross_encoder else None,
                                     chunk_id_map=load_json(BM25_MAP_PATH))
        
    elif strategy == "hybrid":
        entries = hybrid_retrieve(query, embedding_model, normalization_method)
        results = [{"query": query, "results": entries}]
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n검색 결과 (Top-{TOP_K}):")
    for i, r in enumerate(results[0]["results"]):
        print(f"\n⚡ Top {i+1}:\n{r['retrieved_content'].strip()}")
    
    return results
