import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/NEUL77/AI-Engineer/data/merged_qwen3_8b"
PROMPT = "다음 문장을 한국어로 2문장으로 요약해줘:\n대한민국은 21세기 들어 AI와 반도체 산업을 중심으로 글로벌 경쟁력을 강화하고 있다."
MAX_NEW_TOKENS = 128

def measure_hf():
    print("🔄 [HF] 로드 중...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True
    )
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        return_full_text=False,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS
    )
    load_s = time.time() - t0
    print(f"✅ [HF] 로드 {load_s:.2f}s")

    # 워밍업
    _ = gen(PROMPT)

    # 본 측정
    t1 = time.time()
    out = gen(PROMPT)[0]["generated_text"]
    t2 = time.time()
    # 생성 토큰 수 추정
    n_new = len(tok.encode(out))
    tokps = n_new / (t2 - t1)
    print(f"⚙️  [HF] 생성 {t2 - t1:.2f}s, 토큰/초 ≈ {tokps:.2f}")
    return load_s, t2 - t1, tokps

def measure_vllm():
    print("🔄 [vLLM] 로드 중...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        # 초기화 오버헤드 줄이기 위해 필요한 만큼만
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        quantization="bitsandbytes",  # 필요 없다면 제거해 보세요 (초기화 단축 가능)
        trust_remote_code=True,
    )
    load_s = time.time() - t0
    print(f"✅ [vLLM] 로드 {load_s:.2f}s")

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_NEW_TOKENS
    )

    # 워밍업
    _ = llm.generate([PROMPT], sampling)

    # 본 측정
    t1 = time.time()
    outputs = llm.generate([PROMPT], sampling)
    t2 = time.time()
    text = outputs[0].outputs[0].text
    # vLLM은 토크나이저를 안 노출하므로 대략 길이로 추정하거나 HF 토크나이저로 다시 인코딩
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    n_new = len(tok.encode(text))
    tokps = n_new / (t2 - t1)
    print(f"⚙️  [vLLM] 생성 {t2 - t1:.2f}s, 토큰/초 ≈ {tokps:.2f}")
    return load_s, t2 - t1, tokps

if __name__ == "__main__":
    hf_load, hf_gen, hf_tokps = measure_hf()
    print()
    vllm_load, vllm_gen, vllm_tokps = measure_vllm()

    print("\n📊 요약")
    print(f"[HF]   로드 {hf_load:.2f}s | 생성 {hf_gen:.2f}s | tok/s ≈ {hf_tokps:.2f}")
    print(f"[vLLM] 로드 {vllm_load:.2f}s | 생성 {vllm_gen:.2f}s | tok/s ≈ {vllm_tokps:.2f}")