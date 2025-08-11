import os
import re
import json
import shutil
import argparse
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# --- 설정 (기본값으로 사용) ---
DEFAULT_DUMMY_DATA_DIR = "/home/gcp-JeOn-8/RFP_A/data2/output_jsonl"
DEFAULT_CHROMA_DB_DIR = "./chroma_db"  # DB 디렉터리 이름 변경
COLLECTION_NAME = "rfp_documents"      # 컬렉션 이름 변경

# --- 1. 사용할 HuggingFace 모델 지정 ---
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 8

def normalize_source_id(source_id: str) -> str:
    text = source_id.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def load_and_parse_documents(source_dir):
    """
    지정된 디렉터리에서 JSONL 파일을 읽어 LangChain Document 객체 리스트로 변환합니다.
    """
    all_documents = []
    jsonl_files = [f for f in os.listdir(source_dir) if f.endswith(".jsonl")]

    print(f"\n총 {len(jsonl_files)}개의 JSONL 파일(RFP)을 처리합니다.")

    for filename in tqdm(jsonl_files, desc="JSONL 파일 처리 중"):
        file_path = os.path.join(source_dir, filename)
        docs_in_file_count = 0
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        doc_data = json.loads(line)
                        
                        metadata = doc_data.get("metadata", {})
                        
                        source_id = normalize_source_id(metadata.get("source_id", ""))
                        metadata["source_id"] = source_id
                        
                        keywords = metadata.get("keywords", "")
                    
                        if isinstance(keywords, list):
                            keywords = ", ".join(map(str, keywords))

                        metadata["meta_text"] = f"{source_id} {keywords}".strip()
                        
                        doc = Document(
                            page_content=doc_data.get("text", ""), 
                            metadata=metadata  # 수정된 메타데이터를 전달
                        )

                        all_documents.append(doc)
                        docs_in_file_count += 1

                    except json.JSONDecodeError:
                        tqdm.write(f"  - [경고] {filename} 파일의 특정 줄이 유효한 JSON 형식이 아닙니다. 해당 줄은 건너뜁니다.")
            
            if docs_in_file_count > 0:
                tqdm.write(f"  - [성공] {filename} ({docs_in_file_count}개 문서 처리)")
            else:
                tqdm.write(f"  - [정보] {filename} 파일에 처리할 문서 데이터가 없습니다.")

        except Exception as e:
            tqdm.write(f"  - [실패] {filename}: 처리 중 오류 발생 - {e}")

    print(f"\n✅ 총 {len(all_documents)}개의 유효한 문서를 찾았습니다.")
    return all_documents

def add_documents_in_batches(vector_db, documents, batch_size):
    """문서를 배치 단위로 나누어 벡터 DB에 추가합니다."""
    if not documents:
        print("\n⚠️ 처리할 문서가 없습니다.")
        return

    print(f"\n총 {len(documents)}개의 청크를 {batch_size}개씩 나누어 DB에 저장합니다.")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="DB 저장 중"):
        batch = documents[i:i + batch_size]

        for doc in batch:
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    # 리스트의 각 항목을 문자열로 변환하고, 쉼표와 공백으로 연결된 하나의 문자열로 만듭니다.
                    doc.metadata[key] = ", ".join(map(str, value))

        vector_db.add_documents(batch)

    print("\n 모든 문서의 임베딩 및 벡터 DB 저장이 완료되었습니다.")   


def save_chunk_id_mapping(vector_db, save_path = f"/gcp-JeOn-8/RFP_A/data2/chunk_id_map.json"):
    """Chroma 내부 doc.id와 chunk_id를 매핑하여 저장"""
    raw_data = vector_db._collection.get(include=["metadatas"])
    ids = raw_data["ids"]
    metadatas = raw_data["metadatas"]

    mapping = {
        doc_id: metadata.get("chunk_id", "unknown")
        for doc_id, metadata in zip(ids, metadatas)
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"chunk_id 매핑 {len(mapping)}개 저장 완료 → {save_path}")

def run_A(data_dir, db_dir, rebuild):
    """메인 실행 함수"""
    if rebuild and os.path.exists(db_dir):
        print(f"🔄 '{db_dir}' 폴더를 삭제하고 DB를 새로 구축합니다.")
        shutil.rmtree(db_dir)
    os.makedirs(db_dir, exist_ok=True)
    
    # --- 2. HuggingFaceEmbeddings 객체 생성 ---
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}, # E5 모델은 normalize_embeddings를 True로 설정하는 것이 좋습니다.
    )
    
    vector_db = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    doc_count = vector_db._collection.count()

    if doc_count > 0 and not rebuild:
        print(f"\n DB가 이미 '{db_dir}'에 존재하며 {doc_count}개의 문서가 저장되어 있습니다.")
        print("   새로 구축하려면 --rebuild 플래그를 사용하세요.")
        print(f"\n--- [DB 상태 확인] ---")
        print(f"🔍 현재 DB에 저장된 문서 개수: {doc_count}개")
        return  # 함수를 여기서 종료하여 아래의 구축 과정을 생략

    documents = load_and_parse_documents(data_dir)
    add_documents_in_batches(vector_db, documents, BATCH_SIZE)
    save_chunk_id_mapping(vector_db)

    print(f"\n--- [DB 상태 확인] ---")
    count = vector_db._collection.count()
    print(f"🔍 현재 DB에 저장된 문서 개수: {count}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 파일로부터 문서를 임베딩하여 ChromaDB에 저장합니다.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DUMMY_DATA_DIR, help="입력 JSONL 파일이 있는 디렉터리 경로")
    parser.add_argument("--db_dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help="ChromaDB를 저장할 디렉터리 경로")
    parser.add_argument("--rebuild", action="store_true", help="이 플래그를 사용하면 기존 DB를 삭제하고 새로 구축합니다.")
    
    args = parser.parse_args()
    
    run_A(args.data_dir, args.db_dir, args.rebuild)