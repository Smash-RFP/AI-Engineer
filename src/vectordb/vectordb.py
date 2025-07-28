import os
import json

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# openai api key
def check_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print("설정 완료")
    else:
        print("설정 안됨")

# 설정
DUMMY_DATA_DIR = "/home/dlgsueh02/AI-Engineer/data/dummy"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rfp_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# JSON 파일 로드 및 문서 객체 생성 함수
def load_and_parse_documents(source_dir):
    all_documents = []
    json_files = [f for f in os.listdir(DUMMY_DATA_DIR) if f.endswith(".json")]

    print(f"\n총 {len(json_files)}개의 JSON 파일(RFP)을 처리합니다.")

    for filename in json_files:
        file_path = os.path.join(DUMMY_DATA_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 딕셔너리 구조에서 'chunks' 키를 이용해 실제 청크 리스트를 가져옴
                chunks_data = data.get("chunks", []) # .get()을 사용해 'chunks' 키가 없어도 오류 없이 빈 리스트 반환

                if not chunks_data:
                    print(f"  - [경고] {filename} 파일에 처리할 청크 데이터가 없습니다. 건너뜁니다.")
                    continue

                # 각 청크를 LangChain의 Document 객체로 변환
                for chunk in chunks_data:
                    # 'section' 정보를 메타데이터에 추가
                    metadata = {
                        "source": data.get("filename", filename), # JSON 안의 파일명을 우선 사용
                        "chunk_id": chunk.get("chunk_id"),
                        "section": chunk.get("section", "N/A")    # section 정보가 없을 경우 대비
                    }
                    
                    doc = Document(
                        page_content=chunk.get("text", ""), # text가 없는 경우 대비
                        metadata=metadata
                    )
                    all_documents.append(doc)
                
                print(f"  - [성공] {filename} ({len(chunks_data)}개 청크)")

        except json.JSONDecodeError:
            print(f"  - [실패] {filename} 파일이 유효한 JSON 형식이 아닙니다.")
        except Exception as e:
            print(f"  - [실패] {filename} 처리 중 예측하지 못한 오류 발생: {e}")

    print(f"\n총 {len(all_documents)}개의 유효한 문서 조각(청크)을 벡터 DB에 저장합니다.")


#  벡터 DB에 데이터 저장 함수
def add_documents_in_batches(vector_db, all_documents, batch_size):
    if all_documents:
    # 한 번에 처리할 청크 개수 (배치 사이즈)
        batch_size = 100 
        
        print(f"\n총 {len(all_documents)}개의 청크를 {batch_size}개씩 나누어 처리합니다.")

        for i in range(0, len(all_documents), batch_size):
            # all_documents 리스트에서 batch_size만큼 슬라이싱
            batch = all_documents[i:i + batch_size]
            
            # 슬라이싱된 'batch'만 DB에 추가
            vector_db.add_documents(batch)
            
            # 진행 상황 출력
            print(f"  - {i + len(batch)} / {len(all_documents)} 처리 완료")

        print("\n✅ 모든 문서의 임베딩 및 벡터 DB 저장이 완료되었습니다.")

    else:
        print("\n⚠️ 처리할 문서가 없습니다. DUMMY_DATA_DIR 경로를 확인하세요.")


# 구축된 벡터 DB 테스트 함수
def test_vector_db_search(vector_db, query):
    print("\n--- [검색 테스트] ---")
    # 실제 RFP 내용과 관련 있을 법한 질문
    query = "차세대 포털 시스템의 주요 기능 요건은 무엇인가?"

    # retriever가 사용할 유사도 검색 기능 테스트
    # k=3: 가장 유사한 청크 3개를 가져옴
    retrieved_docs = vector_db.similarity_search(query, k=3)

    if retrieved_docs:
        print(f"❓ 질문: \"{query}\"")
        print(f"\n🔍 검색된 유사 청크 Top {len(retrieved_docs)}개:")
        print("-" * 60)
        for i, doc in enumerate(retrieved_docs):
            print(f"[{i+1}] 출처: {doc.metadata.get('source', 'N/A')} (청크 ID: {doc.metadata.get('chunk_id', 'N/A')})")
            print(f"    - 섹션: {doc.metadata.get('section', 'N/A')}")
            print(f"    - 내용: {doc.page_content[:200]}...") # 내용이 기므로 200자만 출력
            print("-" * 60)
    else:
        print("검색된 문서가 없습니다.")

def main():

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    all_documents = load_and_parse_documents(DUMMY_DATA_DIR)

    add_documents_in_batches(vector_db, all_documents, BATCH_SIZE)

    test_query = "차세대 포털 시스템의 주요 기능 요건은 무엇인가?"
    test_vector_db_search(vector_db, test_query)

if __name__ == "__main__":
    check_api_keys()
    main()