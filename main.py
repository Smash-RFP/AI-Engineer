import os
import sys
import json
import shutil
import argparse
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from src.generator.experiment.eval_performance import eval_llm
from src.generator.llm_generator import generate_response
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl
from src.vectordb.vectordb import check_api_keys, load_and_parse_documents, add_documents_in_batches, save_chunk_id_mapping, run

from src.retrieval.modules.bm25_docs_generate import generate_bm25_docs
from src.retrieval.modules.retrieved_contexts import run_retrieve

def process_single_pdf(pdf_path, output_dir, threshold=1.0):
    print(f"🔍 {pdf_path} 처리 중...")
    chunks = extract_text_split_virtual_pages(pdf_path, threshold)
    source_id = sanitize_filename(pdf_path)
    output_path = os.path.join(output_dir, f"{source_id}.jsonl")
    save_chunks_as_jsonl(chunks, source_id, output_path)
    print(f" {source_id}.jsonl 저장 완료! 총 {len(chunks)}개 청크\n")

def run_batch_pipeline(input_dir, output_dir, threshold=1.0):
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(" PDF 파일이 존재하지 않습니다.")
        return
    print(f" 총 {len(pdf_files)}개의 PDF 파일 처리 시작\n")
    for file in tqdm(pdf_files):
        pdf_path = os.path.join(input_dir, file)
        try:
            process_single_pdf(pdf_path, output_dir, threshold)
        except Exception as e:
            print(f" {file} 처리 실패: {e}")

username = "gcp-JeOn"

input_pdf_dir = f"/home/{username}/AI-Engineer/data"
output_jsonl_dir = f"/home/{username}/AI-Engineer/data/dummy"

DEFAULT_DUMMY_DATA_DIR = f"/home/{username}/AI-Engineer/data/dummy"
DEFAULT_CHROMA_DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "rfp_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

#  실행
if __name__ == "__main__":
    run_batch_pipeline(input_pdf_dir, output_jsonl_dir, threshold=1.0)

    parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    args = parser.parse_args()
    
    check_api_keys()
    run(args.data_dir, args.db_dir, args.rebuild)
    
    generate_bm25_docs(
        input_dir=output_jsonl_dir,
        output_pkl_path="data/bm25_docs.pkl",
        output_map_path="data/bm25_chunk_id_map.json"
    )
    
    # retrieval
    QUERY = "해외 지식 재산 센터 사업 관리 시스템 기능 개발 입찰 참가 자격"
    contexts = run_retrieve(QUERY)

    # response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts)
    
    # 대화 이어서 하려면 previous_response_id 파라미터로 넣어줌.
    # response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts, previous_response_id=previous_response_id)

