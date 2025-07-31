import os
import json
import pickle
import re
from langchain_core.documents import Document

chunk_dir = "/home/gcp-JeOn/Smash-RFP/data/dummy_r_1000_100"
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
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # chunks가 없거나 형식이 이상하면 skip
        chunks = data.get("chunks", [])
        if not isinstance(chunks, list):
            print(f"{filename} 처리 실패: 'chunks'가 list 형식이 아님")
            continue

        for i, chunk in enumerate(chunks):
            # 문자열 청크 처리
            if isinstance(chunk, str):
                bm25_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": data.get("filename", filename),
                        "chunk_id": str(i),
                        "source_id": normalize_id(data.get("source_id", filename)),
                    }
                ))
            # dict 청크 처리
            elif isinstance(chunk, dict):
                bm25_docs.append(Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "source": data.get("filename", filename),
                        "chunk_id": chunk.get("chunk_id", str(i)),
                        "source_id": normalize_id(chunk.get("source_id", data.get("source_id", filename))),
                    }
                ))
            else:
                print(f"{filename} - 알 수 없는 chunk 타입: {type(chunk)} → 무시됨")
        
        print(f"{filename} 처리 완료")

    except Exception as e:
        print(f"{filename} 처리 실패: {e}")

# 저장
with open(output_path, "wb") as f:
    pickle.dump(bm25_docs, f)

print(f"\n총 {len(bm25_docs)}개의 Document 객체가 저장되었습니다 → {output_path}")