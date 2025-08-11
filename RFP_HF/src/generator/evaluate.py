from transformers import AutoTokenizer
import json, time, os, re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from langchain_core.messages import HumanMessage, AIMessage
from hf_prompt import get_chat_prompt
from src.retrieval.modules.retrieved_contexts import get_contexts


import os

# openai api key
def check_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print("설정")
    else:
        print("설정 안됨")

if __name__ == "__main__":
    check_api_keys()

client = OpenAI()


# ---------------- 모델 초기화 ----------------
llm = LLM(
    model="/home/NEUL77/AI-Engineer/data/merged_qwen3_8b",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.6,
    quantization="bitsandbytes"
)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=1024
)

tokenizer = AutoTokenizer.from_pretrained("/home/NEUL77/AI-Engineer/data/merged_qwen3_8b")

def trim_to_tokens(text, max_tokens=350):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def build_prompt(query: str, context: str, chat_history=None) -> str:
    if chat_history is None:
        chat_history = []
    prompt_template = get_chat_prompt()
    formatted = prompt_template.format_messages(
        chat_history=chat_history,
        context=context,
        question=query
    )
    return "".join([m.content for m in formatted])

def clean_qwen_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _as_text(item):
    if isinstance(item, dict):
        if isinstance(item.get("retrieved_content"), str):
            return item["retrieved_content"].strip()
        if isinstance(item.get("page_content"), str):
            return item["page_content"].strip()
        for v in item.values():
            if isinstance(v, str) and len(v) > 20:
                return v.strip()
        return ""
    return item.strip() if isinstance(item, str) else ""

def _as_source_chunk(item):
    if isinstance(item, dict):
        src = item.get("retrieved_source_id") or item.get("source_id") or "?"
        chk = item.get("retrieved_chunk_id")  or item.get("chunk_id")  or "?"
        return str(src), str(chk)
    return "?", "?"

# ---------------- 데이터 로드 ----------------
qa_path = "/home/NEUL77/AI-Engineer/data/train/qa_dataset_V3.jsonl"
test_data = [json.loads(l) for l in open(qa_path, "r", encoding="utf-8") if l.strip()]
test_data = test_data[:100]
test_data = [json.loads(l) for l in open(qa_path, "r", encoding="utf-8") if l.strip()]
questions = [ex["Question"].strip() for ex in test_data]
labels    = [ex["Answer"].strip()   for ex in test_data]
print(f"총 {len(test_data)}개 QA 로드 완료")

# ---- 옵션 ----
BATCH = 16
TOP_K = 3
PER_DOC_TOKENS = 350
CKPT_EVERY = 10
CKPT_DIR = "./rag_eval_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------- 평가 루프 ----------------
preds, eval_contexts, all_citations = [], [], []
t0 = time.time()
total_batches = (len(questions) + BATCH - 1) // BATCH

for bi in range(0, len(questions), BATCH):
    batch_idx = bi // BATCH + 1
    qs = questions[bi:bi+BATCH]
    batch_start = time.time()

    merged_ctxs, prompts, cits_batch = [], [], []
    for q in qs:
        retrieved = get_contexts(q, strategy="hybrid", use_cross_encoder=True, use_hyde=False)
        tops = retrieved[:TOP_K] if isinstance(retrieved, list) else []
        ctx_pieces, cits = [], []
        for j, it in enumerate(tops):
            t = trim_to_tokens(_as_text(it), PER_DOC_TOKENS)
            if t:
                ctx_pieces.append(t)
            src, chk = _as_source_chunk(it)
            cits.append(f"[{j+1}] {src}, {chk}")
        merged = "\n\n".join(ctx_pieces)
        full_prompt = build_prompt(q, merged, chat_history=[])
        merged_ctxs.append(merged)
        prompts.append(full_prompt)
        cits_batch.append("\n".join(cits))

    outs = llm.generate(prompts, sampling_params)
    batch_preds = []
    for o in outs:
        text = o.outputs[0].text if (o.outputs and len(o.outputs) > 0) else ""
        text = clean_qwen_response(text).strip() or "해당 질문에 대한 답변을 생성하지 못했습니다."
        batch_preds.append(text)

    preds.extend(batch_preds)
    eval_contexts.extend(merged_ctxs)
    all_citations.extend(cits_batch)

    # 로그 출력
    elapsed = time.time() - t0
    avg_per_batch = elapsed / batch_idx
    remaining = (total_batches - batch_idx) * avg_per_batch
    print(f"[{batch_idx}/{total_batches}] batch {time.time()-batch_start:.1f}s | "
          f"avg {avg_per_batch:.1f}s | ETA {remaining/60:.1f}m | done {len(preds)}/{len(questions)}")

    # 체크포인트 저장
    if batch_idx % CKPT_EVERY == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{batch_idx:03d}.json")
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "preds": preds,
                "eval_contexts": eval_contexts,
                "citations": all_citations,
                "done": len(preds),
                "total": len(questions),
                "avg_sec_per_batch": avg_per_batch
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ 체크포인트 저장: {ckpt_path}")

print(f"생성 완료: preds={len(preds)}, contexts={len(eval_contexts)}, citations={len(all_citations)} | 총 {time.time()-t0:.1f}s")


def evaluate_rag_with_llm(questions, contexts, predictions, labels):
    """
    LLM 평가 메트릭으로 RAG 시스템 평가
    
    Args:
        questions: 질문 리스트
        contexts: 검색 결과 리스트
        predictions: 예측 리스트
        labels: 레이블 리스트
    
    Returns:
        평가 결과가 포함된 데이터프레임
    """
    
    # 결과를 저장할 리스트
    results = []
    
    # 평가 프롬프트
    prompt_template = """
        당신은 RAG(Retrieval-Augmented Generation) 시스템 평가 전문가입니다. 아래 정보를 바탕으로 생성된 답변의 품질을 철저히 평가해주세요.

        질문: {question}

        검색된 컨텍스트:
        {context}

        생성된 답변:
        {prediction}

        참조 답변(정답):
        {label}

        다음 4가지 평가 기준으로 1-5점 척도로 점수를 매겨주세요:

        1. 응답 정확성 (Answer Correctness) [1-5]:
        * 생성된 답변이 참조 답변과 비교하여 정확하고 완전한 정보를 제공하는지 평가
        * 1점: 완전히 잘못된 정보
        * 2점: 부분적으로 관련된 정보를 담고 있으나 대부분 부정확함
        * 3점: 정확한 정보와 부정확한 정보가 혼재되어 있음
        * 4점: 대부분 정확하지만 일부 정보가 누락되거나 미미한 오류가 있음
        * 5점: 참조 답변과 비교했을 때 완전히 정확하고 포괄적인 정보를 제공함

        2. 컨텍스트 관련성 (Context Relevance) [1-5]:
        * 검색된 컨텍스트가 질문에 대답하기 위해 관련성이 높은지 평가
        * 1점: 컨텍스트가 질문과 전혀 관련이 없음
        * 2점: 컨텍스트가 질문과 간접적으로만 관련됨
        * 3점: 컨텍스트 중 일부만 질문과 직접적으로 관련됨
        * 4점: 대부분의 컨텍스트가 질문과 직접적으로 관련됨
        * 5점: 모든 컨텍스트가 질문에 완벽하게 관련되어 있고 불필요한 정보가 없음

        3. 컨텍스트 충실성 (Context Faithfulness) [1-5]:
        * 생성된 답변이 주어진 컨텍스트에만 기반하는지, 아니면 없는 정보를 추가했는지 평가
        * 1점: 답변이 컨텍스트에 없는 정보로만 구성됨 (심각한 환각)
        * 2점: 답변이 주로 컨텍스트에 없는 정보로 구성됨
        * 3점: 답변이 컨텍스트 정보와 없는 정보가 혼합되어 있음
        * 4점: 답변이 주로 컨텍스트에 기반하지만 약간의 추가 정보가 있음
        * 5점: 답변이 전적으로 컨텍스트에 있는 정보만을 사용함

        4. 컨텍스트 충분성 (Context Recall) [1-5]:
        * 검색된 컨텍스트가 질문에 완전히 답변하기에 충분한 정보를 포함하는지 평가
        * 1점: 컨텍스트가 답변에 필요한 정보를 전혀 포함하지 않음
        * 2점: 컨텍스트가 필요한 정보의 일부만 포함함
        * 3점: 컨텍스트가 필요한 정보의 약 절반을 포함함
        * 4점: 컨텍스트가 필요한 정보의 대부분을 포함하지만 일부 누락됨
        * 5점: 컨텍스트가 질문에 완전히 답변하기 위한 모든 필요한 정보를 포함함

        반드시 다음 JSON 형식으로만 응답하세요. 마크다운은 사용하지 않습니다.:
        {
        "answer_correctness": 정수로 된 점수(1-5),
        "context_relevance": 정수로 된 점수(1-5),
        "context_faithfulness": 정수로 된 점수(1-5),
        "context_recall": 점수(1-5),
        "analysis": "종합적인 분석 의견"
        }

        다른 형식의 응답은 하지 마세요. 오직 마크다운이 아닌 JSON만 반환하세요.
        """






# 각 항목에 대해 평가 수행
for i in tqdm(range(len(questions)), total=len(questions), desc="RAG 평가 진행 중"):
    try:
        # 프롬프트 생성 - format 대신 replace 사용
        prompt = prompt_template
        prompt = prompt.replace("{question}", str(questions[i]) if questions[i] is not None else "")
        prompt = prompt.replace("{context}", str(contexts[i]) if contexts[i] is not None else "")
        prompt = prompt.replace("{prediction}", str(predictions[i]) if predictions[i] is not None else "")
        prompt = prompt.replace("{label}", str(labels[i]) if labels[i] is not None else "")

        # GPT-4 API 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 RAG 평가 도구입니다. 반드시 유효한 JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # 결과 파싱
        result = json.loads(response.choices[0].message.content)
        
        # 개별 메트릭 점수 추출
        answer_correctness = result['answer_correctness']
        context_relevance = result['context_relevance']
        context_faithfulness = result['context_faithfulness']
        context_recall = result['context_recall']
        
        # 총점 직접 계산 (개별 메트릭의 합)
        total_score = answer_correctness + context_relevance + context_faithfulness + context_recall
        
        # 원본 데이터와 평가 결과 결합
        row_result = {
            'id': i,
            'question': questions[i],
            'answer_correctness': answer_correctness,
            'context_relevance': context_relevance,
            'context_faithfulness': context_faithfulness,
            'context_recall': context_recall,
            'total_score': total_score,
            'analysis': result['analysis']
        }
        
        results.append(row_result)
        
    except Exception as e:
        print(f"항목 {i} 평가 중 오류 발생: {e}")
        results.append({
            'id': i,
            'question': questions[i],
            'error': str(e)
        })

# 결과 데이터프레임 생성
results_df = pd.DataFrame(results)


# 요약 통계 계산
if 'total_score' in results_df.columns:
    metrics_summary = {
        '평균 총점': results_df['total_score'].mean(),
        '응답 정확성 평균': results_df['answer_correctness'].mean(),
        '컨텍스트 관련성 평균': results_df['context_relevance'].mean(),
        '컨텍스트 충실성 평균': results_df['context_faithfulness'].mean(),
        '컨텍스트 충분성 평균': results_df['context_recall'].mean()
    }
    print("\n===== 평가 요약 =====")
    for metric, value in metrics_summary.items():
        print(f"{metric}: {value:.2f}")

    return results_df, metrics_summary if 'total_score' in results_df.columns else results_df


results_df.to_csv("/home/NEUL77/AI-Engineer/src/generator/results/evaluation_results_01.csv", index=False)



import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib

metric_map = {
    "answer_correctness": "응답 정확성",
    "context_relevance": "컨텍스트 관련성",
    "context_faithfulness": "컨텍스트 충실성",
    "context_recall": "컨텍스트 충분성",
}

# Set1 컬러맵에서 4가지 색상
cmap = plt.get_cmap("Set1", len(metric_map))
bins = np.arange(0.5, 5.6, 1.0)  # 1~5 점수 중앙 정렬

for i, (col_en, label_ko) in enumerate(metric_map.items()):
    plt.figure()
    results_df[col_en].hist(
        bins=bins,
        color=cmap(i),         # Set1에서 색상 지정
        edgecolor="black"
    )
    plt.title(f"{label_ko} 점수 분포")
    plt.xlabel("점수")
    plt.ylabel("개수")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
