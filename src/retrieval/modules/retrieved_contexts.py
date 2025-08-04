import os
from typing import List

from .config import (
    BM25_DOCS_PATH, VECTOR_DB_PATH, TOP_K, HYBRID_ALPHA
)

from .retrieval import (
    generate_hypothetical_passage, retrieve_documents,
    load_vectorstore, load_bm25_documents, init_bm25_retriever,
    hybrid_retrieve, re_rank, load_json
)
from sentence_transformers import CrossEncoder


def get_retriever(strategy: str, top_k: int):
    if strategy == "bm25":
        bm25_docs = load_bm25_documents(BM25_DOCS_PATH)
        return init_bm25_retriever(bm25_docs, k=top_k)
    elif strategy == "dense":
        vectordb = load_vectorstore(persist_dir=VECTOR_DB_PATH)
        return vectordb.as_retriever(search_kwargs={"k": top_k})
    else:
        raise ValueError(f"Invalid retriever strategy: {strategy}")


def get_chunk_id_map(strategy: str):
    if strategy == "bm25":
        return load_json("data/bm25_chunk_id_map.json")
    elif strategy == "dense":
        return load_json("data/chunk_id_map.json")
    else:
        return None


def get_contexts(query: str, strategy: str = "dense", use_cross_encoder: bool = False, top_k: int = TOP_K) -> List[str]:
    # HyDE 쿼리 생성
    hyde_query = generate_hypothetical_passage(query)

    # BM25 + Dense 설정
    if strategy == "hybrid":
        bm25_docs = load_bm25_documents(BM25_DOCS_PATH)
        bm25_retriever = init_bm25_retriever(bm25_docs, k=top_k)
        dense_retriever = load_vectorstore(VECTOR_DB_PATH).as_retriever(search_kwargs={"k": top_k})

        bm25_map = get_chunk_id_map("bm25")
        dense_map = get_chunk_id_map("dense")

        results = hybrid_retrieve(
            bm25_retriever, dense_retriever, [hyde_query],
            alpha=HYBRID_ALPHA,
            bm25_chunk_id_map=bm25_map,
            dense_chunk_id_map=dense_map
        )

        if use_cross_encoder:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            cross_input = [{
                "query": r["query"],
                "retrieved_ids": [
                    {"source_id": e["retrieved_source_id"], "chunk_id": e["retrieved_chunk_id"]}
                    for e in r["results"]
                ],
                "retrieved_contents": [e["retrieved_content"] for e in r["results"]],
                "latency_sec": r["latency_sec"]
            } for r in results]

            reranked = re_rank(cross_input, model=cross_encoder)
            return [r["retrieved_content"] for r in reranked[0]["results"]] if reranked else []

        else:
            return [r["retrieved_content"] for r in results[0]["results"]] if results else []

    # BM25 or Dense (공통 처리)
    else:
        retriever = get_retriever(strategy, top_k)
        chunk_id_map = get_chunk_id_map(strategy)

        if use_cross_encoder:
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            results = retrieve_documents(retriever, [hyde_query], use_rerank=True, model=model, chunk_id_map=chunk_id_map)
        else:
            results = retrieve_documents(retriever, [hyde_query], use_rerank=False, chunk_id_map=chunk_id_map)

        return [r["retrieved_content"] for r in results[0]["results"]] if results else []


def run_retrieve(QUERY):
    # 전략 설정
    contexts = get_contexts(QUERY, strategy="dense", use_cross_encoder=True)
    
    for i, text in enumerate(contexts):
        print(f"\n⚡ Top {i+1}:\n{text.strip()}")
    
    return contexts



