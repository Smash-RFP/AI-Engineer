import os
import argparse
from tqdm import tqdm
from pathlib import Path


from src.generator.llm_generator import generate_response
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl
from src.loader.docling_pdf_processor import run_pdf_pipeline
from src.loader.markdown_chunking_pipeline import run_chunking_pipeline
from src.loader.data_eda import run_eda_pipeline

from src.vectordb.vectordb import check_api_keys, run
from src.retrieval.modules.bm25_docs_generate import generate_bm25_docs
from src.retrieval.modules.retrieved_contexts import run_retrieve

# def process_single_pdf(pdf_path, output_dir, threshold=1.0):
#     print(f" {pdf_path} 처리 중...")
#     chunks = extract_text_split_virtual_pages(pdf_path, threshold)
#     source_id = sanitize_filename(pdf_path)
#     output_path = os.path.join(output_dir, f"{source_id}.jsonl")
#     save_chunks_as_jsonl(chunks, source_id, output_path)
#     print(f" {source_id}.jsonl 저장 완료! 총 {len(chunks)}개 청크\n")

# def run_batch_pipeline(input_dir, output_dir, threshold=1.0):
#     pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
#     if not pdf_files:
#         print(" PDF 파일이 존재하지 않습니다.")
#         return
#     print(f" 총 {len(pdf_files)}개의 PDF 파일 처리 시작\n")
#     for file in tqdm(pdf_files):
#         pdf_path = os.path.join(input_dir, file)
#         try:
#             process_single_pdf(pdf_path, output_dir, threshold)
#         except Exception as e:
#             print(f" {file} 처리 실패: {e}")

# input_pdf_dir = f"/home/{username}/AI-Engineer/data"
# output_jsonl_dir = f"/home/{username}/AI-Engineer/data/dummy"

username = "daeseok"

data_dir = f"/home/{username}/AI-Engineer/data2"
output_docling_dir = f"/home/{username}/AI-Engineer/data2/output_docling"
output_jsonl_dir = Path(f"/home/{username}/AI-Engineer/data2/output_jsonl")
output_eda_dir = f"/home/{username}/AI-Engineer/data2/output_eda"

pdf_trigger = Path(output_docling_dir) / "pdf_processed.flag"
chunk_trigger = Path(output_jsonl_dir) / "chunk_processed.flag"


DEFAULT_DUMMY_DATA_DIR = f"/home/{username}/AI-Engineer/data/dummy"
DEFAULT_CHROMA_DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "rfp_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

def continue_response(QUERY:str, previous_response_id=None):
    run_retrieve(QUERY)
    contexts = run_retrieve(QUERY)
    response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts, previous_response_id=previous_response_id)
    print('response_text: ', response_text)

    return response_text, previous_response_id

#  실행
if __name__ == "__main__":
    
    # run_batch_pipeline(input_pdf_dir, output_jsonl_dir, threshold=1.0)     
    run_eda_pipeline(data_dir, output_eda_dir , output_jsonl_dir)
    
    if not pdf_trigger.exists():
        print(" PDF 파이프라인 실행 중...")
        run_pdf_pipeline(input_dir=data_dir, output_dir=output_docling_dir)
        pdf_trigger.touch()
    else:
        print(" PDF 파이프라인은 이미 처리됨. 건너뜀.")

    #  Markdown → JSONL 청킹
    if not chunk_trigger.exists():
        print(" Markdown 청킹 파이프라인 실행 중...")
        run_chunking_pipeline(root_dir=Path(output_docling_dir), output_dir=Path(output_jsonl_dir))
        chunk_trigger.touch()
    else:
        print(" 청킹 파이프라인은 이미 처리됨. 건너뜀.")    

    # parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    # parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    # parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    # parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    # args = parser.parse_args()
    
    # check_api_keys()
    # run(args.data_dir, args.db_dir, args.rebuild)
    
    # generate_bm25_docs(
    #     input_dir=output_jsonl_dir,
    #     output_pkl_path="data/bm25_docs.pkl",
    #     output_map_path="data/bm25_chunk_id_map.json"
    # )
    
    # # retrieval
    # QUERY = "국민연금공단이 발주한 이러닝시스템 관련 사업 추진표에 대해 알려줘"
    # run_retrieve(QUERY)
    # contexts = run_retrieve(QUERY)

    # response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts)
    # print('response_text: ', response_text)
    
    # """
    # 대화를 이어하는 방법
    # 1. Query 변수 값 변경
    # ex)

    # response_text, previous_response_id = continue_response("콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘." , previous_response_id)
    # response_text, previous_response_id = continue_response("교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?" , previous_response_id)
    # ...
    
    # """
    # response_text, previous_response_id = continue_response("콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘." , previous_response_id)
    # response_text, previous_response_id = continue_response("교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?" , previous_response_id)