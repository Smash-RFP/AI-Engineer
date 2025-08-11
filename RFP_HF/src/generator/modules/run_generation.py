import sys
sys.path.append("/home/NEUL77/AI-Engineer/src/retrieval/modules")

import os
import time
from typing import List, Dict, Union
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage

from retrieval import (
    load_vectorstore,
    load_bm25_documents,
    init_bm25_retriever,
    retrieve_documents,
    hybrid_retrieve,
    re_rank
)
from config import TOP_K, HYBRID_ALPHA
from prompt_template import few_shot_prompt 
import openai_generation as openai_generation

# 텔레메트리 비활성화
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_retriever_results(
    query: str,
    retriever_type: str,
    db_path: str,
    bm25_docs_path: str,
    k: int,
    alpha: float,
    use_rerank: bool = False
):
    """
    주어진 쿼리에 대해 지정된 리트리버 전략을 사용하여 관련 문서를 검색하고,
    Cross-Encoder 기반 reranking을 적용하는 함수
    """
    start_retrieval = time.perf_counter()

    vectordb = load_vectorstore(db_path)
    if vectordb._collection.count() > 0:
        print(f"Vectorstore 문서 개수: {vectordb._collection.count()}개") 
    dense_ret = vectordb.as_retriever(search_kwargs={"k": k})

    if retriever_type == "dense":
        temp = retrieve_documents(dense_ret, [query])
    elif retriever_type == "bm25":
        docs = load_bm25_documents(bm25_docs_path)
        bm25_ret = init_bm25_retriever(docs, k=k)
        temp = retrieve_documents(bm25_ret, [query])
    elif retriever_type == "hybrid":
        docs = load_bm25_documents(bm25_docs_path)
        bm25_ret = init_bm25_retriever(docs, k=k)
        temp = hybrid_retrieve(bm25_ret, dense_ret, [query], alpha=alpha)
    else:
        raise ValueError(f"지원하지 않는 retriever_type입니다: {retriever_type}")
    
    retrieval_latency = time.perf_counter() - start_retrieval
    
    if use_rerank:
        if not temp or "results" not in temp[0] or not temp[0]["results"]:
            print("[re_rank] 검색된 문서가 없어 re-rank를 수행하지 않습니다.")
            return temp

        entries = temp[0]["results"]
        ids = [
            {"source_id": e.get("retrieved_source_id", "unknown"),
            "chunk_id": e.get("retrieved_chunk_id", "unknown")}
            for e in entries
        ]
        contents = [e.get("retrieved_content", "") for e in entries]

        rerank_start = time.perf_counter()
        reranked_results = re_rank([{
            "query": temp[0]["query"],
            "retrieved_ids": ids,
            "retrieved_contents": contents,
            "latency_sec": retrieval_latency
        }], cross_encoder_model, top_k=k)
        rerank_latency = time.perf_counter() - rerank_start

        temp = reranked_results
        temp[0]["latency_sec"] += rerank_latency
        print(f"[re_rank] 입력 문서 수: {len(contents)} -> 최종 문서 수: {len(temp[0]['results'])}")

    first = temp[0]
    entries = first.get("results", [])
    retrieved_ids = [e.get("retrieved_source_id", "unknown") for e in entries]
    retrieved_filenames = [e.get("retrieved_chunk_id", "unknown") for e in entries]
    retrieved_contents = [e.get("retrieved_content", "") for e in entries]
    
    return [{
        "query": first["query"],
        "retrieved_ids": retrieved_ids[:k],
        "retrieved_filenames": retrieved_filenames[:k],
        "retrieved_contents": retrieved_contents[:k],
        "latency_sec": first.get("latency_sec", 0.0)
    }]


def wrap_retrieval_results(raw_results: List[dict]) -> List[Document]:
    """
    검색 결과(raw_results)를 LangChain Document 객체로 변환하여 반환하는 함수
    """
    if not raw_results:
        return []
        
    first = raw_results[0]
    ids = first.get("retrieved_ids", [])
    filenames = first.get("retrieved_filenames", [])
    contents = first.get("retrieved_contents", [])
    
    min_len = min(len(ids), len(filenames), len(contents))
    
    metas = [
        {"source": fname, "source_id": doc_id}
        for fname, doc_id in zip(filenames[:min_len], ids[:min_len])
    ]
    return [
        Document(page_content=cont, metadata=meta)
        for cont, meta in zip(contents[:min_len], metas)
    ]

def print_answer_result(query: str, answer: str, docs: List[Document]):
    """
    질문, LLM이 생성한 답변, 사용된 문서들의 출처 정보를 출력하는 함수
    """
    print("\n[질문]:", query, flush=True)
    print("\n[답변]:\n", answer, flush=True)
    print("\n[출처 문서]:", flush=True)
    if docs:
        for i, doc in enumerate(docs, start=1):
            filename = doc.metadata.get("source_id", "N/A")
            chunk_id = doc.metadata.get("source", "N/A")
            print(f"  [{i}] {filename}, {chunk_id}")
    else:
        print("  답변에 활용된 문서가 없습니다.")

def postprocess_answer(answer: str) -> str:
    """
    중복된 답변이 2번 출력되는 현상을 방지하기 위한 후처리 함수.
    동일한 문단이 반복될 경우 첫 번째만 남김.
    """
    lines = answer.strip().split("\n\n")
    seen = set()
    unique_lines = []
    for line in lines:
        if line.strip() and line.strip() not in seen:
            unique_lines.append(line)
            seen.add(line.strip())
    return "\n\n".join(unique_lines)

def run_generation(
    query: str,
    chat_history: List[Union[HumanMessage, AIMessage]],
    retriever_type: str = "dense",
    model_name: str = "gpt-4o",
    db_path: str = "/home/NEUL77/NEUL/data/chroma_db",
    bm25_docs_path: str = "/home/NEUL77/NEUL/data/bm25_docs.pkl",
    k: int = 5,
    alpha: float = 0.5,
    verbose: bool = False,
    use_rerank: bool = False
):
    """
    RAG 시스템의 end-to-end 실행 함수로, 쿼리에 대해 문서를 검색하고,
    LLM을 통해 응답을 생성하며, 결과를 출력하는 함수 (후속 질문 처리 기능 추가)
    """
    start_total = time.perf_counter()

    llm = openai_generation.get_llm_openai(model_name=model_name)

    # 쿼리 재작성 프롬프트 수정: 검색 쿼리만을 생성하도록 명확하게 지시
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "대화 기록을 바탕으로 후속 질문을 독립적인 검색 쿼리로 재작성하세요. 절대 추가적인 텍스트나 답변을 제공하지 말고, 오직 검색에 필요한 쿼리 문장만 제공해야 합니다. 검색 쿼리가 원본 질문과 동일하다면 원본 질문을 그대로 사용하세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    contextualized_query_chain = contextualize_q_prompt | llm | StrOutputParser()
    contextualized_query = contextualized_query_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    
    if verbose:
        print(f"Original Query: {query}")
        print(f"Contextualized Query: {contextualized_query}")

    # 재작성된 쿼리로 문서 검색
    raw = get_retriever_results(
        contextualized_query, 
        retriever_type,
        db_path,
        bm25_docs_path,
        k,
        alpha,
        use_rerank=use_rerank
    )

    if verbose:
        print(f"★★★ [{retriever_type.upper()}] → {len(raw[0]['retrieved_ids'])}개 문서 획득 (k={k}, α={alpha})")

    # LangChain Document 객체로 래핑 및 context 생성
    docs_for_qa = wrap_retrieval_results(raw)
    context_string = "\n\n".join([doc.page_content for doc in docs_for_qa])

    # QA 체인 실행
    prompt = few_shot_prompt(verbose=verbose)
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "question": query, 
        "context": context_string,
        "chat_history": chat_history
    })

    answer = postprocess_answer(answer)

    end_total = time.perf_counter()
    total_latency = end_total - start_total

    print_answer_result(query, answer, docs_for_qa)

    return {
        "result": answer,
        "source_documents": docs_for_qa,
        "latency_sec": round(total_latency, 4),
        "retriever_type": retriever_type
    }

if __name__ == "__main__":
    questions = [
        "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.",
        "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.",
        "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
        "기초과학연구원이 발주한 극저온시스템 관련 사업에서 AI 기반 예측에 대한 요구사항이 있나?",
        "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.",
        "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
        "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?"
    ]
    
    chat_history = []
    
    for i, q in enumerate(questions, start=1):
        print(f"\n▶▶▶ Q&A {i} ◀◀◀")
        
        result = run_generation(
            query=q,
            chat_history=chat_history,
            retriever_type="dense",
            model_name="gpt-4o",
            db_path="/home/NEUL77/NEUL/data/chroma_db",
            bm25_docs_path="/home/NEUL77/NEUL/data/bm25_docs.pkl",
            k=TOP_K,
            alpha=HYBRID_ALPHA,
            verbose=True,
            use_rerank=True
        )
        
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=result["result"]))
        
        print("-" * 100)