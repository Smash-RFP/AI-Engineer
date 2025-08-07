import os
import re
import json
import pickle
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

# ======== 설정 ========
from src.retrieval.modules.embedding_loader import load_embedding_model

def normalize_source_id(text: str) -> str:
    text = text.strip().replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def generate_meta_embeddings(data_dir: str, save_path: str, model_name="text-embedding-3-small", provider="openai"):
    embedding_model = load_embedding_model(model_name, provider)
    meta_embedding_dict = {}

    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    print(f"\n총 {len(jsonl_files)}개의 JSONL 파일을 처리합니다.")

    for filename in tqdm(jsonl_files, desc="meta_embedding 생성 중"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    doc_data = json.loads(line)
                    metadata = doc_data.get("metadata", {})

                    source_id = normalize_source_id(metadata.get("source_id", ""))
                    chunk_id = metadata.get("chunk_id", "unknown")
                    try:
                        chunk_id = int(chunk_id)
                    except:
                        chunk_id = "unknown"

                    full_id = f"{source_id}_{chunk_id}"
                    if full_id in meta_embedding_dict:
                        continue

                    keywords = metadata.get("keywords", metadata.get("key_word", []))
                    if isinstance(keywords, list):
                        keywords = ", ".join(map(str, keywords))
                    elif not isinstance(keywords, str):
                        keywords = str(keywords)

                    meta_text = f"{source_id} {keywords}".strip()
                    if not meta_text:
                        continue

                    embedding = embedding_model.embed_query(meta_text)
                    meta_embedding_dict[full_id] = embedding

                except Exception as e:
                    tqdm.write(f"[경고] {filename} line {line_num} 처리 실패: {e}")
                    continue


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(meta_embedding_dict, f)

    print(f"\nmeta_embedding_dict 저장 완료 → {save_path}")
    print(f"총 {len(meta_embedding_dict)}개 저장됨. 예시 키: {list(meta_embedding_dict.keys())[:5]}")


if __name__ == "__main__":
    generate_meta_embeddings(DEFAULT_DATA_DIR, DEFAULT_SAVE_PATH)
