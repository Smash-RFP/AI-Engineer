import re
import sys
sys.path.append("/home/gcp-JeOn-8/RFP_A")
from vllm import LLM, SamplingParams
from langchain_core.messages import HumanMessage, AIMessage
from src.generator.hf_prompt import get_chat_prompt
from src.retrieval.retrieval_run import retrieved_contexts


def build_prompt(query: str, context: str, chat_history: list) -> str:

    """
    LangChain의 ChatPromptTemplate을 수동으로 풀어주는 함수.
    """
    print("build_prompt 실행 시작")
    prompt_template = get_chat_prompt()
    formatted = prompt_template.format_messages(
        chat_history=chat_history,
        context=context,
        question=query
    )
    print(prompt_template)
    return "".join([msg.content for msg in formatted])

# 후처리 함수
def clean_qwen_response(generated_text: str) -> str:
    # <think> ~ </think> 제거
    if "<think>" in generated_text and "</think>" in generated_text:
        return generated_text.split("</think>")[-1].strip()
    return generated_text.strip()

chat_history = []

def run_rag_with_vllm(query: str, llm, sampling_params) -> str:
    global chat_history
    # chat_history.clear()

    # Retrieve
    contexts = retrieved_contexts(query, model_name="BAAI/bge-m3", provider="huggingface")
    merged_context = "\n\n".join([
    doc.get("retrieved_content", doc.get("page_content", "")).strip()
    for doc in contexts
])


    # Prompt 생성
    full_prompt = build_prompt(query, merged_context, chat_history)
    print("full_prompt 정의됨")

    # vLLM 생성
    outputs = llm.generate([full_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    # chat history 업데이트
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))

    # citation 구성
    citations = []
    for i, doc in enumerate(contexts):
        source_id = doc.get("retrieved_source_id", "unknown")
        chunk_id = doc.get("retrieved_chunk_id", "unknown")
        citations.append(f"[{i+1}] {source_id}, {chunk_id}")

    return clean_qwen_response(response).rstrip() + "\n\n" + "\n".join(citations)


# def run_rag_with_vllm(query: str, llm, sampling_params) -> str:
#     global chat_history

#     # 1) Retrieve
#     contexts = retrieved_contexts(query, model_name="BAAI/bge-m3", provider="huggingface") or []

#     # 2) 컨텍스트 병합 (dict/str 안전)
#     merged_context = "\n\n".join(t for t in (_extract_text(d) for d in contexts) if t)

#     # 3) Prompt 생성
#     full_prompt = build_prompt(query, merged_context, chat_history)

#     # 4) vLLM 생성
#     outputs = llm.generate([full_prompt], sampling_params)
#     response = outputs[0].outputs[0].text.strip()

#     # 5) chat history
#     chat_history.append(HumanMessage(content=query))
#     chat_history.append(AIMessage(content=response))

#     # 6) citations (dict/str 안전)
#     citations = [_extract_citation(doc, i+1) for i, doc in enumerate(contexts)]

#     return clean_qwen_response(response).rstrip() + ("\n\n" + "\n".join(citations) if citations else "")


# 마크다운 제거 함수
def remove_markdown_formatting(text: str) -> str:
    # 굵은 글씨 제거
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 제목 스타일 제거
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


def _extract_text(doc):
    """dict/str 모두에서 텍스트만 안전하게 뽑기"""
    if isinstance(doc, dict):
        for k in ("retrieved_content", "page_content", "content", "text"):
            v = doc.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(doc).strip()  # 이미 문자열이거나 기타 타입


def _extract_citation(doc, idx):
    # dict 형식
    if isinstance(doc, dict):
        # metadata 안에서 읽기
        if "metadata" in doc and isinstance(doc["metadata"], dict):
            src = doc["metadata"].get("source_id", "unknown")
            cid = doc["metadata"].get("chunk_id", "unknown")
        else:
            src = doc.get("retrieved_source_id", "unknown")
            cid = doc.get("retrieved_chunk_id", "unknown")
        return f"[{idx}] {src}, {cid}"
    return f"[{idx}] unknown, unknown"



# if __name__ == "__main__":
#     queries = [
#         "부산국제영화제의 온라인 서비스 재개발 및 유지관리 사업에서는 어떤 작업 유형을 포함하나요?",
#         "2024년도 평택시 버스정보시스템 구축사업의 주요 목표와 기대효과는 무엇인가요?",
#         "버스정류소 안내단말기 소프트웨어의 주요 기능은 무엇인가요?",
#         "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
#     ]

#     # LLM 로드 
#     llm = LLM(
#         model="/home/NEUL77/AI-Engineer/data/merged_qwen3_8b",
#         dtype="float16",
#         max_model_len=8192,
#         gpu_memory_utilization=0.75,
#         quantization="bitsandbytes"
#     )
#     sampling_params = SamplingParams(
#         temperature=0.2,
#         top_p=0.9,
#         max_tokens=512
#     )

#     # 쿼리 순차 처리
#     for query in queries:
#         response = run_rag_with_vllm(query)  # 내부에서 llm.generate 사용
#         print(remove_markdown_formatting(response))
#         print("-" * 80)