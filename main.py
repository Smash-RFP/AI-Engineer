import os
import argparse
from tqdm import tqdm

from src.generator.llm_generator import generate_response
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl
from src.vectordb.vectordb import check_api_keys, run

from src.retrieval.modules.bm25_docs_generate import generate_bm25_docs
from src.retrieval.modules.retrieved_contexts import run_retrieve

def is_preprocessing_complete(input_pdf_dir, output_jsonl_dir):
    pdf_files = [f for f in os.listdir(input_pdf_dir) if f.lower().endswith(".pdf")]

    processed_files = set(
        os.path.splitext(f)[0] for f in os.listdir(output_jsonl_dir) if f.endswith(".jsonl")
    )

    def sanitize(filename):
        return re.sub(r"[^\w]+", "_", os.path.splitext(filename)[0])

    sanitized_pdf_names = {sanitize(f) for f in pdf_files}

    return sanitized_pdf_names.issubset(processed_files)

username = "daeseok"

input_pdf_dir = f"/home/{username}/AI-Engineer/data"
output_jsonl_dir = f"/home/{username}/AI-Engineer/data/dummy"

DEFAULT_DUMMY_DATA_DIR = f"/home/{username}/AI-Engineer/data/dummy"
DEFAULT_CHROMA_DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "rfp_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

#  실행
if __name__ == "__main__":   
    
    os.makedirs(output_jsonl_dir, exist_ok=True)

    if is_preprocessing_complete(input_pdf_dir, output_jsonl_dir):
        print("모든 PDF가 전처리되어 있습니다. 전처리 스킵.")
    else:
        print("일부 PDF가 전처리되지 않았습니다. 처리 시작합니다.\n")
        pdf_files = [f for f in os.listdir(input_pdf_dir) if f.lower().endswith(".pdf")]

        def sanitize(filename):
            return re.sub(r"[^\w]+", "_", os.path.splitext(filename)[0])

        processed_files = set(
            os.path.splitext(f)[0] for f in os.listdir(output_jsonl_dir) if f.endswith(".jsonl")
        )
        
        for file in tqdm(pdf_files):
            if sanitize(file) in processed_files:
                continue 
            try:
                pdf_path = os.path.join(input_pdf_dir, file)
                process_single_pdf(pdf_path, output_jsonl_dir)
            except Exception as e:
                print(f" {file} 처리 실패: {e}")
 

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
    run_retrieve(QUERY)
    contexts = run_retrieve(QUERY)

    response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts)
    
    # 대화 이어서 하려면 previous_response_id 파라미터로 넣어줌.
    # response_text, previous_response_id = generate_response(query=QUERY, retrieved_rfp_text=contexts, previous_response_id=previous_response_id)  