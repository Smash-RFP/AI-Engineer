import os
import json
import pickle
import re
from langchain_core.documents import Document

chunk_dir = "/home/gcp-JeOn/Smash-RFP/data/dummy/"
output_path = "../data/bm25_docs.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

bm25_docs = []

# 정규화 함수 정의
def normalize_id(text: str) -> str:
    text = text.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

for filename in os.listdir(chunk_dir):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(chunk_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        # case: {"chunks": [...]}
        if isinstance(data, dict) and "chunks" in data:
            for chunk in data["chunks"]:
                # source_id를 normalize
                raw_source_id = chunk.get("source_id") or filename
                normalized_source_id = normalize_id(raw_source_id)

                bm25_docs.append(Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "chunk_id": chunk.get("chunk_id", ""),
                        "source_id": normalized_source_id
                    }
                ))

# 저장
with open(output_path, "wb") as f:
    pickle.dump(bm25_docs, f)

print(f"{len(bm25_docs)}개의 청크가 저장되었습니다 → {output_path}")
