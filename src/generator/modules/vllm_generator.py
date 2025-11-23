import sys, re
sys.path.append("/home/NEUL77/AI-Engineer")

from vllm import LLM, SamplingParams
from langchain_core.messages import HumanMessage, AIMessage
from hf_prompt import get_chat_prompt
from src.retrieval.retrieval_run import retrieved_contexts

# ---- 설정 ----
EMBEDDING_MODEL = "nlpai-lab/KoE5"  # 임베딩 모델명(리트리버용)

def build_prompt(query: str, context: str, chat_history: list) -> str:
    prompt_template = get_chat_prompt()
    formatted = prompt_template.format_messages(
        chat_history=chat_history,
        context=context,
        question=query
    )
    return "".join([msg.content for msg in formatted])

# <think> 제거
def clean_qwen_response(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

# 리트리브 결과
def _to_text(d):
    if hasattr(d, "page_content"):
        return (d.page_content or "").strip()
    if isinstance(d, dict):
        return (d.get("retrieved_content") or d.get("page_content") or "").strip()
    return ""

def _to_meta(d):
    if hasattr(d, "metadata"):
        md = d.metadata or {}
        sid = md.get("source_id") or md.get("retrieved_source_id") or "unknown"
        cid = md.get("chunk_id")  or md.get("retrieved_chunk_id")  or "unknown"
        return sid, cid
    if isinstance(d, dict):
        md = d.get("metadata") or {}
        sid = d.get("retrieved_source_id") or md.get("source_id") or "unknown"
        cid = d.get("retrieved_chunk_id") or md.get("chunk_id")  or "unknown"
        return sid, cid
    return "unknown", "unknown"

chat_history = []

def run_rag_with_vllm(query: str) -> str:
    global chat_history

    # Retrieve
    contexts = retrieved_contexts(query, model_name=EMBEDDING_MODEL, provider="huggingface")

    # 컨텍스트 병합 + citation 구성 (상위 5개만 사용)
    snippets, citations = [], []
    for i, doc in enumerate(contexts[:5]):
        txt = _to_text(doc)
        if not txt:
            continue
        snippets.append(txt)
        sid, cid = _to_meta(doc)
        citations.append(f"[{i+1}] {sid}, {cid}")

    merged_context = "\n\n".join(snippets)

    # Prompt 생성
    full_prompt = build_prompt(query, merged_context, chat_history)

    # 생성
    outputs = llm.generate([full_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    # 히스토리 업데이트
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))

    # 후처리 + citation 붙이기
    answer = clean_qwen_response(response).rstrip()
    if citations:
        answer += "\n\n" + "\n".join(citations)
    return answer

# 마크다운 제거
def remove_markdown_formatting(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text

if __name__ == "__main__":
    queries = [
        "부산국제영화제의 온라인 서비스 재개발 및 유지관리 사업에서는 어떤 작업 유형을 포함하나요?",
        "2024년도 평택시 버스정보시스템 구축사업의 주요 목표와 기대효과는 무엇인가요?",
        "버스정류소 안내단말기 소프트웨어의 주요 기능은 무엇인가요?",
        "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
    ]

    # LLM 로드 (vLLM)
    llm = LLM(
        model="/home/NEUL77/AI-Engineer/data/merged_qwen3_8b",
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        quantization="bitsandbytes",
    )
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        stop=["</think>"],                 # 체인오브소트 출력 방지
        include_stop_str_in_output=False,
    )

    for q in queries:
        resp = run_rag_with_vllm(q)
        print(remove_markdown_formatting(resp))
        print("-" * 80)
