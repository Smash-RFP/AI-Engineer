import os
import json
import pickle
import re
from langchain_core.documents import Document

def normalize_id(text: str) -> str:
    text = text.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def generate_bm25_docs(input_dir: str, output_pkl_path: str, output_map_path: str):
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    bm25_docs = []
    chunk_id_map = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    metadata = record.get("metadata", {})
                    source_id = metadata.get("source_id", f"{filename}_line{line_num}")
                    chunk_id = metadata.get("chunk_id", f"chunk-{line_num:04d}")
                    
                    identifier = f"{source_id}_chunk_{chunk_id}"
                    chunk_id_map[identifier] = {
                        "source_id": source_id,
                        "chunk_id": chunk_id
                    }

                    bm25_docs.append(Document(
                        page_content=text,
                        metadata={
                            "source_id": source_id,
                            "chunk_id": chunk_id
                        }
                    ))
                except Exception as e:
                    print(f"{filename} line {line_num} 처리 실패: {e}")
                    continue

    print(f"총 {len(bm25_docs)}개의 Document가 생성됨. 저장 중...")

    with open(output_map_path, "w", encoding="utf-8") as f:
        json.dump(chunk_id_map, f, ensure_ascii=False, indent=2)

    with open(output_pkl_path, "wb") as f:
        pickle.dump(bm25_docs, f)

    print(f"저장 완료 → {output_pkl_path}")
