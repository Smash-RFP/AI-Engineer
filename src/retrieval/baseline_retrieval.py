# retrieval - baseline

import os
import json
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain.retrievers import ContextualCompressionRetriever

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# 데이터 로드
def load_vectorstore(persist_dir: str) -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding, 
        collection_name='rfp_documents'
        )
    return vectordb


# retrieval 정의 - top-k + 중복 제외
def run_baseline_retrieval(queries: List[str], vectordb: Chroma, k: int = 3) -> List[Dict]:
    retriever = vectordb.as_retriever(search_kwargs={"k": k * 3})  # 중복 제거를 위해 k보다 많이 가져오기
    results = []

    for query in queries:
        raw_docs = retriever.get_relevant_documents(query)

        # 중복 제거: page_content 기준
        seen_contents = set()
        unique_docs = []
        for doc in raw_docs:
            content_key = doc.page_content.strip()
            if content_key not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_key)
            if len(unique_docs) == k:
                break

        results.append({
            "query": query,
            "retrieved_docs": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content[:500]
                }
                for doc in unique_docs
            ]
        })
    return results


# 결과 저장
def save_results(results: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        

queries = [
    "요구사항 출입관리시스템 공급",
    "중이온 가속기용 극저온시스템 목적",
    "고려대학교 차세대 포털·학사 정보시스템 구축 사업"
  ]

vectordb = load_vectorstore("../vectordb/chroma_db")
results = run_baseline_retrieval(queries, vectordb)
save_results(results, "./results/baseline_results.json")