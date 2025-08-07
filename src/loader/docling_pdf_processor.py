import logging
import time
import sys
sys.setrecursionlimit(5000)
from pathlib import Path
from enum import Enum
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- 이미지 저장 방식 지정 ---
class ImageRefMode(str, Enum):
    EMBEDDED = "embedded" # 이미지 내용을 markdown 내부에 포함
    REFERENCED = "referenced"  # 이미지 파일 경로로 참조

# --- 출력 이미지 해상도 배율 (2.0 = 약 144 DPI) ---
IMAGE_RESOLUTION_SCALE = 2.0

def process_pdf(input_doc_path: Path, output_root_dir: Path, doc_converter: DocumentConverter):
    """
    단일 PDF 파일을 처리하여 다음 파일을 생성:
    - 페이지 이미지: page-0.png, page-1.png, ...
    - 테이블 이미지: table-1.png, ...
    - 그림 이미지: picture-1.png, ...
    - 마크다운 파일: -with-images.md (inline), -with-image-refs.md (ref link)

    Args:
        input_doc_path (Path): 입력 PDF 경로
        output_root_dir (Path): 출력 파일을 저장할 루트 폴더
        doc_converter (DocumentConverter): Docling 변환기 인스턴스
    """        
    logging.info(f" 처리 중: {input_doc_path.name}")
    start_time = time.time()

    doc_filename = input_doc_path.stem
    output_dir = output_root_dir / doc_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    conv_res = doc_converter.convert(input_doc_path)

    for page_no, page in conv_res.document.pages.items():
        page_image_filename = output_dir / f"{doc_filename}-page-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    table_counter = 0
    picture_counter = 0
    for element, _ in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
            with filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
        elif isinstance(element, PictureItem):
            picture_counter += 1
            filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            with filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    md_embed = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_embed, image_mode=ImageRefMode.EMBEDDED)

    md_refs = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_refs, image_mode=ImageRefMode.REFERENCED)

    elapsed = time.time() - start_time
    logging.info(f"{input_doc_path.name} 완료 (소요 시간: {elapsed:.2f}초)")

def run_pdf_pipeline(input_dir: str = "data", output_dir: str = "output_docling"):
    """
    지정된 디렉토리 내의 모든 PDF 파일을 변환하여,
    - 페이지 이미지
    - 테이블 및 그림 이미지
    - 마크다운 (inline + ref) 파일 생성

    Args:
        input_dir (str): PDF 파일이 위치한 입력 디렉토리
        output_dir (str): 결과물을 저장할 출력 디렉토리
    """    
    logging.basicConfig(level=logging.INFO)

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    pdf_files = list(input_root.glob("*.pdf"))
    logging.info(f"{len(pdf_files)}개의 PDF 파일 발견됨")

    for pdf_path in pdf_files:
        process_pdf(pdf_path, output_root, doc_converter)