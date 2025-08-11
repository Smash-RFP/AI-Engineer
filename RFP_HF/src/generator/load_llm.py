# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# model_name = "/home/NEUL77/AI-Engineer/data/merged_qwen3_8b"

# # 4bit 양자화 설정
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# 4bit로 모델 로딩
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",  
#     quantization_config=bnb_config,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16 
# )

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipelin

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

if __name__ == "__main__":
    llm = load_qwen_llm()