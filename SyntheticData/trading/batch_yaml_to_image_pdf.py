import os
import io
import yaml
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import fitz          # PyMuPDF  (pip install PyMuPDF)
from PIL import Image # Pillow   (pip install Pillow)


# ──────────────────────────────────────────────
# 1. 한글 폰트 등록
# ──────────────────────────────────────────────
def register_korean_font() -> str:
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            name = "CustomKoreanFont"
            pdfmetrics.registerFont(TTFont(name, path))
            return name
    print("경고: 한글 폰트 미검출 → 기본 폰트 사용 (한글 깨짐 가능)")
    return "Helvetica"


# ──────────────────────────────────────────────
# 2. YAML → ReportLab Story (기존 로직 유지)
# ──────────────────────────────────────────────
def parse_yaml_to_story(data, story, style, level=0):
    indent = "&nbsp;" * (level * 4)

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                story.append(Paragraph(f"{indent}<b>{key}</b>:", style))
                parse_yaml_to_story(value, story, style, level + 1)
            else:
                story.append(Paragraph(f"{indent}<b>{key}</b>: {value}", style))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                story.append(Paragraph(f"{indent}<b>[ Item {idx + 1} ]</b>", style))
                parse_yaml_to_story(item, story, style, level + 1)
            elif isinstance(item, list):
                parse_yaml_to_story(item, story, style, level + 1)
            else:
                story.append(Paragraph(f"{indent}• {item}", style))
    else:
        story.append(Paragraph(f"{indent}{data}", style))


# ──────────────────────────────────────────────
# 3. 텍스트 PDF를 "메모리"에 생성
# ──────────────────────────────────────────────
def build_text_pdf_bytes(data, font_name: str) -> bytes:
    """ReportLab으로 텍스트 PDF를 만들어 bytes 반환"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)

    normal_style = ParagraphStyle(
        "NormalStyle",
        fontName=font_name,
        fontSize=10,
        leading=16,
        spaceAfter=6,
    )

    story = []
    parse_yaml_to_story(data, story, normal_style)
    doc.build(story)

    return buf.getvalue()


# ──────────────────────────────────────────────
# 4. 텍스트 PDF → 고해상도 이미지 리스트
# ──────────────────────────────────────────────
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """PyMuPDF로 각 페이지를 렌더링하여 PIL Image 리스트 반환"""
    images: list[Image.Image] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    zoom = dpi / 72  # 72 dpi가 PDF 기본 단위
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


# ──────────────────────────────────────────────
# 5. 이미지 리스트 → 이미지 전용 PDF 저장
# ──────────────────────────────────────────────
def save_images_as_pdf(images: list[Image.Image], output_path: Path):
    """Pillow로 이미지들을 하나의 PDF로 저장 (텍스트 레이어 없음)"""
    if not images:
        raise ValueError("저장할 이미지가 없습니다.")

    first = images[0]
    rest  = images[1:] if len(images) > 1 else []

    first.save(
        str(output_path),
        save_all=True,
        append_images=rest,
        resolution=200.0,   # PDF 메타데이터상 DPI
        format="PDF",
    )


# ──────────────────────────────────────────────
# 6. 파이프라인: YAML → 이미지 PDF
# ──────────────────────────────────────────────
def create_image_pdf(data, pdf_path: Path, font_name: str, dpi: int = 200):
    """
    YAML dict → 텍스트 PDF(bytes) → 이미지 변환 → 이미지 PDF 저장
    최종 파일에는 텍스트 레이어가 전혀 없음
    """
    pdf_bytes   = build_text_pdf_bytes(data, font_name)
    images      = pdf_bytes_to_images(pdf_bytes, dpi=dpi)
    save_images_as_pdf(images, pdf_path)


# ──────────────────────────────────────────────
# 7. 배치 처리
# ──────────────────────────────────────────────
def process_batch(input_dir: str, output_dir: str, dpi: int = 200):
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    font_name = register_korean_font()

    yaml_files = sorted(
        list(in_path.glob("*.yaml")) + list(in_path.glob("*.yml"))
    )

    if not yaml_files:
        print(f"'{input_dir}' 폴더에 처리할 YAML 파일이 없습니다.")
        return

    print(f"총 {len(yaml_files)}개 파일 발견. 이미지 PDF 변환 시작 (DPI={dpi})...\n")

    success, fail = 0, 0

    for yf in yaml_files:
        pdf_file = out_path / f"{yf.stem}.pdf"
        try:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                print(f"[스킵] {yf.name} (빈 파일)")
                continue

            create_image_pdf(data, pdf_file, font_name, dpi=dpi)
            print(f"[성공] {yf.name}  →  {pdf_file.name}")
            success += 1

        except Exception as e:
            print(f"[실패] {yf.name}  오류: {e}")
            fail += 1

    print("\n" + "=" * 40)
    print(f"완료!  성공 {success}건 / 실패 {fail}건")
    print("=" * 40)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FOLDER  = "input_yamls"
    OUTPUT_FOLDER = "output_pdfs"
    DPI           = 200          # 150~300 사이 권장 (높을수록 선명, 파일 크기 ↑)

    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, dpi=DPI)