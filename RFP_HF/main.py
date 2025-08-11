import os
import argparse
from tqdm import tqdm

from src.generator.llm_generator import generate_response
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl
from src.loader.docling_pdf_processor import run_pdf_pipeline
from src.loader.markdown_chunking_pipeline import run_chunking_pipeline
from src.loader.data_eda import run_eda_pipeline
from src.generator.vllmhf import run_rag_with_vllm

from src.vectordb.vectordbA import run_A
from src.vectordb.vectordb import check_api_keys, run_B
from src.vectordb.bm25_docs_generate_AB import generate_bm25_docs
from src.vectordb.meta_embedding_generate_A import generate_meta_embeddings
from src.retrieval.retrieval_run import retrieved_contexts
from src.generator.vllmhf import run_rag_with_vllm
from src.generator.vllmhf import remove_markdown_formatting

from vllm import LLM, SamplingParams
from src.generate.hf_generator import load_qwen_llm

username = "gcp-JeOn-8"

base_dir = f"/home/{username}/RFP_A"
data_dir = os.path.join(base_dir, "data2")
output_docling_dir = os.path.join(data_dir, "output_docling")
output_jsonl_dir = os.path.join(data_dir, "output_jsonl")
output_eda_dir = os.path.join(data_dir, "output_eda")

pdf_trigger = os.path.join(output_docling_dir, "pdf_processed.flag")
chunk_trigger = os.path.join(output_jsonl_dir, "chunk_processed.flag")

DEFAULT_DUMMY_DATA_DIR = os.path.join(base_dir, "data2", "output_jsonl")
DEFAULT_CHROMA_DB_DIR = f"/home/{username}/RFP_A/data2/chroma_db"
DEFAULT_SAVE_PATH = f"/home/{username}/RFP_A/data2/meta_embedding_dict.pkl"
COLLECTION_NAME = "rfp_documents"

def continue_response(QUERY: str, previous_response_id=None):
    run_retrieve(QUERY)
    contexts = run_retrieve(QUERY)
    response_text, previous_response_id = generate_response(
        query=QUERY, retrieved_rfp_text=contexts, previous_response_id=previous_response_id
    )
    print('response_text: ', response_text)
    return response_text, previous_response_id


def openai_llm_response(user_query: str, previous_response_id=None, model: str = "gpt-4.1-nano", embedding_model="text-embedding-3-small"):
    contexts = retrieved_contexts(user_query, model_name=embedding_model, provider="openai")
    response_text, previous_response_id = generate_response(
        query=user_query, retrieved_rfp_text=contexts, previous_response_id=previous_response_id, model=model
    )
    print('response_text: ', response_text)
    return response_text, previous_response_id


def huggingface_llm_response(user_query: str, previous_response_id=None, model: str = "gpt-4o-nano", embedding_model="nlpai-lab/KoE5"):
    contexts = retrieved_contexts(user_query, model_name=embedding_model, provider="huggingface")
    return "response_text"  # 수정 필요 시 구현


def pipeline(user_query: str, previous_response_id=None, model: str = "gpt-4o-nano"):

    # docling 전처리
    
    # if not os.path.exists(pdf_trigger):
    #     print(" PDF 파이프라인 실행 중...")
    #     run_pdf_pipeline(input_dir=data_dir, output_dir=output_docling_dir)
    #     with open(pdf_trigger, 'w') as f:
    #         f.write('')
    # else:
    #     print(" PDF 파이프라인은 이미 처리됨. 건너뜀.")

    # # Markdown → JSONL 청킹
    # if not os.path.exists(chunk_trigger):
    #     print(" Markdown 청킹 파이프라인 실행 중...")
    #     run_chunking_pipeline(root_dir=output_docling_dir, output_dir=output_jsonl_dir)
    #     with open(chunk_trigger, 'w') as f:
    #         f.write('')
    # else:
    #     print(" 청킹 파이프라인은 이미 처리됨. 건너뜀.")

    ## 시나리오 B
    ## embedding_B
     
    # EMBEDDING_MODEL = "text-embedding-3-small"
    # parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    # parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    # parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    # parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    # args = parser.parse_args()
    
    # check_api_keys()
    # run_B(args.data_dir, args.db_dir, args.rebuild)
    
    
    # generate_bm25_docs(
    #         input_dir=output_jsonl_dir,
    #         output_pkl_path="data2/bm25_docs.pkl",
    #         output_map_path="data2/bm25_chunk_id_map.json"
    #     )
    
    # generate_meta_embeddings(data_dir=DEFAULT_DUMMY_DATA_DIR, save_path=DEFAULT_SAVE_PATH, model_name=EMBEDDING_MODEL, provider="openai")
    #contexts = retrieved_contexts(query=user_query, model_name=EMBEDDING_MODEL, provider="openai")
    
    ## 시나리오 A
    ## embedding_A    
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    args = parser.parse_args()        
    run_A(args.data_dir, args.db_dir, args.rebuild)
    
    # generate_bm25_docs(
    #         input_dir=output_jsonl_dir,
    #         output_pkl_path= f"/home/{username}/RFP_A/data2/bm25_docs.pkl",
    #         output_map_path= f"/home/{username}/RFP_A/data2/bm25_chunk_id_map.json"
    #     )
    
    # generate_meta_embeddings(data_dir=DEFAULT_DUMMY_DATA_DIR, save_path=DEFAULT_SAVE_PATH, model_name=EMBEDDING_MODEL, provider="huggingface")
    
    contexts = retrieved_contexts(query=user_query, model_name=EMBEDDING_MODEL, provider="huggingface")

    # LLM 로드 
    # llm = LLM(
    #     model="/home/NEUL77/AI-Engineer/data/merged_qwen3_8b",
    #     dtype="float16",
    #     max_model_len=8192,
    #     gpu_memory_utilization=0.75,
    #     quantization="bitsandbytes"
    # )
    
    MODEL_DIR = "/home/NEUL77/AI-Engineer/data/merged_qwen3_8b"
    llm = load_qwen_llm(model_dir=MODEL_DIR, max_new_tokens=1024)
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024
    )   
    
    #remove_markdown_formatting(run_rag_with_vllm(user_query))
    
    response = run_rag_with_vllm(user_query,llm,sampling_params)
    print(remove_markdown_formatting(response))
    
    

if __name__ == "__main__":
    pipeline("고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?", None, "test_model")    

