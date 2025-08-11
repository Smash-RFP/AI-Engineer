import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from hf_prompt import get_chat_prompt
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retrieval_run import retrieved_contexts
from langchain_core.messages import HumanMessage, AIMessage

MODEL_DIR = "/home/NEUL77/AI-Engineer/data/merged_qwen3_8b"

def load_qwen_llm(model_dir=MODEL_DIR, max_new_tokens=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    text_gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=text_gen_pipe)

# 후처리 함수
def clean_qwen_response(generated_text: str) -> str:
    # <think> ~ </think> 제거
    if "<think>" in generated_text and "</think>" in generated_text:
        return generated_text.split("</think>")[-1].strip()
    return generated_text.strip()

chat_history = []

def run_rag_with_qwen(query: str) -> str:
    global chat_history

    # 리트리버 문서
    contexts = retrieved_contexts(query=user_query, model_name=EMBEDDING_MODEL, provider="huggingface")
    merged_context = "\n\n".join([doc["retrieved_content"].strip() for doc in contexts])

    # rag chain
    raw_response = rag_chain.invoke({
        "question": query,
        "context": merged_context,
        "chat_history": chat_history
    })

    # chat history 업로드
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=raw_response))

    # 출처문서
    citations = []
    for i, doc in enumerate(contexts):
        source_id = doc.get("retrieved_source_id", "unknown")
        chunk_id = doc.get("retrieved_chunk_id", "unknown")
        citations.append(f"[{i+1}] {source_id}, {chunk_id}")

    return clean_qwen_response(raw_response).rstrip() + "\n\n" + "\n".join(citations)

# 마크다운 제거 함수
def remove_markdown_formatting(text: str) -> str:
    # 굵은 글씨 제거
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 제목 스타일 제거
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text

if __name__ == "__main__":
    prompt = get_chat_prompt()
    llm = load_qwen_llm()
    rag_chain = prompt | llm | StrOutputParser()

    queries = [
        "부산국제영화제의 온라인 서비스 재개발 및 유지관리 사업에서는 어떤 작업 유형을 포함하나요?",
        "2024년도 평택시 버스정보시스템 구축사업의 주요 목표와 기대효과는 무엇인가요?",
        "버스정류소 안내단말기 소프트웨어의 주요 기능은 무엇인가요?",
        "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
    ]

    for query in queries:
        response = run_rag_with_qwen(query)
        print(remove_markdown_formatting(response))
        print("-" * 80)