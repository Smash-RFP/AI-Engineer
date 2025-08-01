
import os
from tqdm import tqdm
from src.loader.preprocessing import extract_text_split_virtual_pages, sanitize_filename, save_chunks_as_jsonl

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

#  실행
if __name__ == "__main__":
    input_pdf_dir = "/home/daeseok/AI-Engineer/data"
    output_jsonl_dir = "/home/daeseok/AI-Engineer/data/dummy"
    run_batch_pipeline(input_pdf_dir, output_jsonl_dir, threshold=1.0)