import os
import argparse
from tqdm import tqdm

from src.generator.llm_generator import generate_response
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl
from src.loader.docling_pdf_processor import run_pdf_pipeline
from src.loader.markdown_chunking_pipeline import run_chunking_pipeline
from src.loader.data_eda import run_eda_pipeline

from src.vectordb.vectordb import check_api_keys, run
from src.retrieval.modules.bm25_docs_generate import generate_bm25_docs
from src.retrieval.modules.retrieved_contexts import run_retrieve


username = "dlgsueh02"

base_dir = f"/home/{username}/AI-Engineer"
data_dir = os.path.join(base_dir, "data2")
output_docling_dir = os.path.join(data_dir, "output_docling")
output_jsonl_dir = os.path.join(data_dir, "output_jsonl")
output_eda_dir = os.path.join(data_dir, "output_eda")

pdf_trigger = os.path.join(output_docling_dir, "pdf_processed.flag")
chunk_trigger = os.path.join(output_jsonl_dir, "chunk_processed.flag")

DEFAULT_DUMMY_DATA_DIR = os.path.join(base_dir, "data2", "output_jsonl")
DEFAULT_CHROMA_DB_DIR = "./data2/chroma_db"
COLLECTION_NAME = "rfp_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def continue_response(QUERY: str, previous_response_id=None):
    run_retrieve(QUERY)
    contexts = run_retrieve(QUERY)
    response_text, previous_response_id = generate_response(
        query=QUERY, retrieved_rfp_text=contexts, previous_response_id=previous_response_id
    )
    print('response_text: ', response_text)
    return response_text, previous_response_id

def openai_llm_response(user_query: str, previous_response_id=None, model: str = "gpt-4.1-nano"):
    run_retrieve(user_query)
    contexts = run_retrieve(user_query)
    response_text, previous_response_id = generate_response(
        query=user_query, retrieved_rfp_text=contexts, previous_response_id=previous_response_id, model=model
    )
    print('response_text: ', response_text)
    return response_text, previous_response_id

def huggingface_llm_response(user_query: str, previous_response_id=None, model: str = "gpt-4o-nano"):
    run_retrieve(user_query)
    contexts = run_retrieve(user_query)
    return "response_text"  # 수정 필요 시 구현


def pipeline(user_query: str, previous_response_id=None, model: str = "gpt-4o-nano"):

    if not os.path.exists(pdf_trigger):
        print("📄 PDF 파이프라인 실행 중...")
        run_pdf_pipeline(input_dir=data_dir, output_dir=output_docling_dir)
        with open(pdf_trigger, 'w') as f:
            f.write('')
    else:
        print("📄 PDF 파이프라인은 이미 처리됨. 건너뜀.")

    # Markdown → JSONL 청킹
    if not os.path.exists(chunk_trigger):
        print("🔍 Markdown 청킹 파이프라인 실행 중...")
        run_chunking_pipeline(root_dir=output_docling_dir, output_dir=output_jsonl_dir)
        with open(chunk_trigger, 'w') as f:
            f.write('')
    else:
        print("🔍 청킹 파이프라인은 이미 처리됨. 건너뜀.")

    parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    args = parser.parse_args()
    
    check_api_keys()
    run(args.data_dir, args.db_dir, args.rebuild)


if __name__ == "__main__":
    pipeline("hello", None, "test_model")
