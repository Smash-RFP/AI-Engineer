import os
import json
import pickle
from langchain_core.documents import Document

chunk_dir = "/home/gcp-JeOn/Smash-RFP/data/dummy/"
output_path = "../data/bm25_docs.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

bm25_docs = []

bm25_docs = []
for filename in os.listdir(chunk_dir):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(chunk_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        # case: {"chunks": [...]}
        if isinstance(data, dict) and "chunks" in data:
            for chunk in data["chunks"]:
                bm25_docs.append(Document(
                    page_content=chunk["text"],
                    metadata={"chunk_id": chunk.get("chunk_id", ""), "source_id": filename}
                ))

# 저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(bm25_docs, f)

print(f"{len(bm25_docs)}개의 청크가 저장되었습니다.")

print(f"총 {len(bm25_docs)}개의 청크가 저장되었습니다 → {output_path}")
