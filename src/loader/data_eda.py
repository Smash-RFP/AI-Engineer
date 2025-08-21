import os
import re
import glob
import json
import fitz
import pdfplumber
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from tqdm import tqdm


def setup_korean_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if not os.path.exists(font_path):
        print("⚠️ 한글 폰트가 없습니다. 기본 폰트로 진행합니다.")
        return
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = font_prop.get_name()
    mpl.rcParams['axes.unicode_minus'] = False


def count_pages_adjusting_for_two_column(pdf_path, threshold_ratio=1.0):
    try:
        doc = fitz.open(pdf_path)
        real_page_count = 0
        has_two_column_page = False
        for page in doc:
            ratio = page.rect.width / page.rect.height
            if ratio > threshold_ratio:
                real_page_count += 2
                has_two_column_page = True
            else:
                real_page_count += 1
        return real_page_count, has_two_column_page
    except Exception as e:
        print(f" 페이지 수 오류: {pdf_path} - {e}")
        return 0, False


def extract_text_and_count_chars(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return sum(len(page.get_text()) for page in doc)
    except Exception as e:
        print(f" 글자 수 오류: {pdf_path} - {e}")
        return 0


def classify_rfp(filename, category_map):
    name = filename.replace(".pdf", "").strip()
    parts = name.split("_")
    org = parts[0].strip() if len(parts) > 1 else "미상"
    large = next((category for keyword, category in category_map.items() if keyword in org), "기타")
    year_match = re.search(r"(20\d{2})", name)
    year = year_match.group(1) if year_match else "미상"
    keyword_candidates = ["고도화", "구축", "운영", "개선", "컨설팅", "유지보수", "재구축", "기능개선", "시스템", "플랫폼", "ERP", "ISP"]
    keywords = ", ".join([kw for kw in keyword_candidates if kw in name]) or "기타"
    return {"대분류": large, "중분류(기관명)": org, "연도": year, "사업 키워드": keywords, "파일명": name}


def analyze_table_counts(pdf_paths):
    results = []
    for pdf_path in tqdm(pdf_paths, desc="📄 테이블 수 분석 중"):
        file = os.path.basename(pdf_path)
        table_count = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    table_count += len(page.extract_tables())
        except Exception as e:
            print(f"⚠️ {file} 처리 중 오류: {e}")
            continue
        results.append({"filename": file, "num_tables": table_count})
    return pd.DataFrame(results)


def analyze_jsonl_chunks(jsonl_dir, output_dir):
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    if not jsonl_files:
        print("⚠️ JSONL 파일이 없습니다.")
        return

    total_chunks = 0
    total_keywords = 0
    file_count = 0
    chunks_per_file = []

    for jsonl_file in tqdm(jsonl_files, desc="📂 JSONL 청크 분석 중"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
                chunk_count = len(chunks)
                keyword_count = sum(len(chunk.get("metadata", {}).get("key_word", [])) for chunk in chunks)

                total_chunks += chunk_count
                total_keywords += keyword_count
                chunks_per_file.append(chunk_count)
                file_count += 1
        except Exception as e:
            print(f"⚠️ 오류 발생: {jsonl_file} - {e}")
            continue

    if file_count == 0:
        print("❌ 유효한 JSONL 파일 없음")
        return

    avg_chunks_per_file = total_chunks / file_count
    avg_keywords_per_chunk = total_keywords / total_chunks if total_chunks > 0 else 0

    print("\n📊 JSONL 청크 통계 요약")
    print(f" 총 JSONL 파일 수: {file_count}")
    print(f" 평균 청크 수 (파일당): {avg_chunks_per_file:.2f}")
    print(f" 평균 키워드 수 (청크당): {avg_keywords_per_chunk:.2f}")

    # 🔽 청크 수 분포 히스토그램
    plt.figure(figsize=(10, 6))
    bins = range(0, max(chunks_per_file) + 5, 5)
    plt.hist(chunks_per_file, bins=bins, color="mediumpurple", edgecolor="black", rwidth=0.9)
    plt.title("JSONL 파일당 청크 수 분포", fontsize=14)
    plt.xlabel("청크 수", fontsize=12)
    plt.ylabel("파일 수", fontsize=12)
    plt.xticks(bins)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jsonl_청크수_히스토그램.png"))
    plt.close()


def run_eda_pipeline(pdf_dir, output_dir, jsonl_dir=None):
    setup_korean_font()
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)

    # 실질 페이지 수 및 글자 수 분석
    adjusted_page_counts = []
    char_counts = []
    two_column_pdf_count = 0
    for path in tqdm(pdf_files, desc="📄 실질 페이지/글자 수 분석"):
        pages, has_two_col = count_pages_adjusting_for_two_column(path)
        chars = extract_text_and_count_chars(path)
        adjusted_page_counts.append(pages)
        char_counts.append(chars)
        if has_two_col:
            two_column_pdf_count += 1

    print("\n📊 PDF 통계 요약")
    print(f" 총 PDF 파일 수: {len(pdf_files)}")
    print(f" 평균 실질 페이지 수 (2단 포함): {np.mean(adjusted_page_counts):.2f}")
    print(f" 평균 글자 수: {np.mean(char_counts):.2f}")
    print(f" 2단 구성 PDF 수: {two_column_pdf_count}")
    print(f" 최대 글자 수: {np.max(char_counts)}")
    print(f" 최소 글자 수: {np.min(char_counts)}")

    # 히스토그램 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(adjusted_page_counts, bins=20, color='skyblue', edgecolor='black')
    plt.title("PDF 페이지 수 분포 (2단 보정 포함)")
    plt.xlabel("실질 페이지 수")
    plt.ylabel("파일 수")

    plt.subplot(1, 2, 2)
    plt.hist(char_counts, bins=20, color='lightgreen', edgecolor='black')
    plt.title("PDF 글자 수 분포")
    plt.xlabel("글자 수")
    plt.ylabel("파일 수")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "페이지_글자수_히스토그램.png"))
    plt.close()

    # 파일명 기반 RFP 분류
    category_map = {
        "대학교": "대학교", "대학": "대학교", "학교": "대학교",
        "공사": "공기업", "공단": "공기업",
        "재단법인": "공공기관", "재단": "공공기관",
        "협회": "협회/단체", "진흥원": "진흥기관",
        "지원센터": "지원센터", "연구원": "연구기관",
        "위원회": "정부기관", "청": "정부기관", "부": "정부기관",
        "광역시": "지자체", "특별시": "지자체", "시청": "지자체", "도청": "지자체", "군청": "지자체"
    }

    rfp_files = [os.path.basename(f) for f in pdf_files]
    df_rfp = pd.DataFrame([classify_rfp(f, category_map) for f in rfp_files])

    # 저장
    rfp_output_path = os.path.join(output_dir, "RFP_분류_결과.xlsx")
    df_rfp.to_excel(rfp_output_path, index=False)
    print(f"\n✅ RFP 분류 결과 저장: {rfp_output_path}")
    print(df_rfp.to_string(index=False))

    # 분포 시각화 저장
    distribution_table = pd.crosstab(df_rfp["사업 키워드"], df_rfp["대분류"])
    top_keywords = distribution_table.sum(axis=1).sort_values(ascending=False).head(10).index
    top_distribution = distribution_table.loc[top_keywords]

    plt.figure(figsize=(12, 6))
    top_distribution.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 6))
    plt.title("상위 10개 RFP 사업 키워드의 대분류별 분포")
    plt.xlabel("사업 키워드")
    plt.ylabel("RFP 건수")
    plt.xticks(rotation=45)
    plt.legend(title="기관 대분류", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RFP_키워드_분포.png"))
    plt.close()

    # 테이블 수 분석
    df_tables = analyze_table_counts(pdf_files)
    print(f"\n📈 평균 테이블 수: {df_tables['num_tables'].mean():.2f}")

    bins = range(0, df_tables["num_tables"].max() + 20, 20)
    plt.figure(figsize=(10, 6))
    plt.hist(df_tables["num_tables"], bins=bins, color="salmon", edgecolor="black", rwidth=0.9)
    plt.title("PDF별 테이블 수 분포 (20개 단위 구간)", fontsize=14)
    plt.xlabel("PDF 내 테이블 수 범위", fontsize=12)
    plt.ylabel("PDF 파일 개수", fontsize=12)
    plt.xticks(bins)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "테이블_수_분포.png"))
    plt.close()

    # JSONL 분석 (옵션)
    if jsonl_dir:
        analyze_jsonl_chunks(jsonl_dir, output_dir)