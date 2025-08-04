import os
import json
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_source_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return re.sub(r"[^\w]", "_", base)

def process_pdf_directory(data_dir: str, output_dir: str, limit: int = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"총 파일 수: {len(os.listdir(data_dir))}")
    print(f"PDF 파일 수: {len(pdf_files)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)

            source_id = generate_source_id(filename)
            filename_without_ext = os.path.splitext(filename)[0]

            result = {
                "source_id": source_id,
                "chunks": [
                    {
                        "chunk_id": i,
                        "text": chunk.page_content
                    }
                    for i, chunk in enumerate(chunks)
                ]
            }

            output_path = os.path.join(output_dir, f"{filename_without_ext}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"✅ 처리 완료: {filename}")
        except Exception as e:
            print(f"❌ 오류 발생: {filename} → {e}")            
            
            

# from processor.pdf_chunking import process_pdf_directory

# def main():
#     data_dir = "/home/daeseok/AI-Engineer/data"
#     output_dir = "/home/daeseok/AI-Engineer/data/prepared_data"
#     process_pdf_directory(data_dir, output_dir, limit=4)

# if __name__ == "__main__":
#     main()