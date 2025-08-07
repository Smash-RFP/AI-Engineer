import os
import re
import json
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
import fnmatch

# --- 반복된 문구 제거 (특정 패턴 기반) ---
def remove_repeated_phrases(text):
    def is_table_or_ascii_art(line):
        return (
            line.strip().startswith("|") or
            re.match(r'^[\|\-‖═│╭╮╯╰]+$', line.strip()) or
            any(s in line for s in ["‖", "──", "│", "━"])
        )

    def is_image_path(line):
        return re.match(r'^Image.*\.(png|jpg|jpeg|gif)\)?$', line.strip())

    def clean_line(line):
        if is_table_or_ascii_art(line) or is_image_path(line):
            return line
        line = re.sub(r'\b(\w+)( \1\b)+', r'\1', line)
        line = re.sub(r'\b(\d+)( \1\b)+', r'\1', line)
        line = re.sub(r'(\b[가-힣]{2,}\b)( \1)+', r'\1', line)
        line = re.sub(r'([가-힣])\s+\1(\s+\1)*', r'\1', line)
        line = re.sub(r'((?:[가-힣a-zA-Z0-9]+\s+){1,5}[가-힣a-zA-Z0-9]+)(?:\s+\1)+', r'\1', line)
        return line

    lines = text.split('\n')
    cleaned_lines = [clean_line(line) for line in lines]
    return '\n'.join(cleaned_lines)

def remove_repeats_top_n_pages(pages: List[str], n=3) -> List[str]:
    return [
        remove_repeated_phrases(p) if i < n else p
        for i, p in enumerate(pages)
    ]

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\s\W]+', '_', filename)

def clean_source_id(raw_name: str) -> str:
    cleaned = (
        raw_name
        .replace("-with-image-refs", "")
        .replace("_with_image_refs_md", "")
        .replace("_with_image_refs", "")
    )
    return sanitize_filename(cleaned)

def extract_chunks_by_double_sharp(lines: List[str]) -> List[Dict]:
    pattern = r'^##\s+(.+)'
    chunks = []
    current_chunk = []
    current_keywords = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_chunk:
                chunks.append({
                    "lines": current_chunk,
                    "keywords": list(dict.fromkeys(current_keywords))
                })
                current_chunk = []
                current_keywords = []
            current_keywords.append(match.group(1).strip())
        current_chunk.append(line)

    if current_chunk:
        chunks.append({
            "lines": current_chunk,
            "keywords": list(dict.fromkeys(current_keywords))
        })
    return chunks

def merge_small_chunks_min400(chunks: List[Dict], min_len: int = 400) -> List[Dict]:
    merged = []
    buffer = {"lines": [], "keywords": []}
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        chunk_text = '\n'.join(chunk["lines"]).strip()

        if not buffer["lines"]:
            buffer["lines"] = chunk["lines"]
            buffer["keywords"] = chunk["keywords"]
            i += 1
            continue

        buffer_text = '\n'.join(buffer["lines"]).strip()
        if len(buffer_text) < min_len:
            buffer["lines"].extend(chunk["lines"])
            buffer["keywords"].extend(chunk["keywords"])
            i += 1
        else:
            merged.append({
                "text": buffer_text,
                "key_word": list(dict.fromkeys(buffer["keywords"]))
            })
            buffer = {"lines": [], "keywords": []}
    if buffer["lines"]:
        merged.append({
            "text": '\n'.join(buffer["lines"]).strip(),
            "key_word": list(dict.fromkeys(buffer["keywords"]))
        })
    return merged

def full_markdown_chunking(text: str, file_name: str) -> List[Dict]:
    source_id = clean_source_id(file_name)
    pages = text.split('\f')
    cleaned = remove_repeats_top_n_pages(pages, n=3)
    text = '\n'.join(cleaned)

    lines = text.split('\n')
    raw_chunks = extract_chunks_by_double_sharp(lines)
    merged_chunks = merge_small_chunks_min400(raw_chunks, min_len=400)

    final_chunks = []
    for idx, chunk in enumerate(merged_chunks):
        final_chunks.append({
            "text": chunk["text"],
            "source_id": source_id,
            "key_word": chunk["key_word"]
        })
    return final_chunks

def clean_keywords(keywords):
    cleaned = []
    for kw in keywords:
        kw = re.sub(r"&[a-z]+;", " ", kw)
        kw = re.sub(r"[^가-힣a-zA-Z\s]", "", kw)
        kw = kw.strip()
        if kw:
            cleaned.append(kw)
    return list(dict.fromkeys(cleaned))

def convert_chunks_with_id_per_source(chunks_by_source):
    result = {}
    for raw_source_id, chunk_list in chunks_by_source.items():
        clean_source = clean_source_id(raw_source_id)
        result[clean_source] = []
        for idx, ch in enumerate(chunk_list):
            clean_kw = clean_keywords(ch["key_word"])
            chunk_no = idx + 1
            result[clean_source].append({
                "id": f"{clean_source}-{chunk_no}",
                "text": ch["text"],
                "metadata": {
                    "source_id": clean_source,
                    "key_word": clean_kw,
                    "chunk_id": chunk_no
                }
            })
    return result

def run_chunking_pipeline(root_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    chunks_by_source_id = defaultdict(list)

    # 파일 탐색: "-with-image-refs.md"만 찾기
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, "*-with-image-refs.md"):
            md_path = os.path.join(dirpath, filename)
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                raw_file_name = os.path.splitext(filename)[0]
                source_id = clean_source_id(raw_file_name)
                chunks = full_markdown_chunking(content, source_id)
                for ch in chunks:
                    ch["source_id"] = source_id
                    chunks_by_source_id[source_id].append(ch)

    converted_chunks_dict = convert_chunks_with_id_per_source(chunks_by_source_id)

    for source_id, chunk_list in converted_chunks_dict.items():
        file_name = f"{source_id}.jsonl"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in chunk_list:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        print(f" 저장 완료: {file_path} ({len(chunk_list)}개 청크)")