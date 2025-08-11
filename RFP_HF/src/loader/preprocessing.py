import os
import re
import json
import pdfplumber

def is_two_column_by_width(page, threshold=1.0):
    return (page.width / page.height) > threshold

def clean_text(text):
    if text is None:
        return ""
    lines = text.split('\n')
    return "\n".join([line.strip() for line in lines])

def convert_table_to_markdown(table_data):
    if not table_data:
        return ""
    max_cols = max(len(row) for row in table_data)
    header_row = "|" + "|".join(clean_text(cell) for cell in table_data[0]) + "|"
    separator_row = "|" + "|".join(["---"] * len(table_data[0])) + "|"
    data_rows = []
    for row in table_data[1:]:
        padded_row = row + [""] * (max_cols - len(row))
        data_rows.append("|" + "|".join(clean_text(cell) for cell in padded_row) + "|")
    return "\n".join([header_row, separator_row] + data_rows)

def sanitize_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^\w]+", "_", name)

def extract_text_split_virtual_pages(pdf_path, threshold=1.0):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        virtual_page_num = 1
        for page_idx, page in enumerate(pdf.pages):
            width, height = page.width, page.height
            if is_two_column_by_width(page, threshold):
                for crop_region in [(0, width / 2), (width / 2, width)]:
                    cropped = page.crop((crop_region[0], 0, crop_region[1], height))
                    text = cropped.extract_text() or ""
                    tables = cropped.extract_tables()
                    md_tables = "\n\n".join([convert_table_to_markdown(t) for t in tables if t])
                    content = text.strip()
                    if md_tables:
                        content += "\n\n[Table]\n" + md_tables
                    results.append(content)
                    virtual_page_num += 1
            else:
                text = page.extract_text() or ""
                tables = page.extract_tables()
                md_tables = "\n\n".join([convert_table_to_markdown(t) for t in tables if t])
                content = text.strip()
                if md_tables:
                    content += "\n\n[Table]\n" + md_tables
                results.append(content)
                virtual_page_num += 1
    return results

def save_chunks_as_jsonl(chunks, source_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            json.dump({
                "text": chunk,
                "metadata": {
                    "source_id": source_id,
                    "chunk_id": f"chunk-{idx:04d}"
                }
            }, f, ensure_ascii=False)
            f.write("\n")