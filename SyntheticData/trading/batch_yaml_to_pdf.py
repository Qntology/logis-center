import os
import yaml
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_korean_font():
    """OS별 한글 폰트 등록 (환경에 맞춰 경로 수정 필요)"""
    # 윈도우: 맑은 고딕, 맥: 애플 고딕, 리눅스: 나눔고딕 등 환경에 맞게 지정하세요.
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",                     # Windows
        "/System/Library/Fonts/AppleGothic.ttf",           # Mac
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux (Ubuntu)
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            font_name = 'CustomKoreanFont'
            pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
            
    print("경고: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. (한글 깨짐 발생 가능)")
    return 'Helvetica'

def create_pdf(data, pdf_path, font_name):
    """단일 딕셔너리 데이터를 PDF로 생성"""
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []

    # 폰트 스타일 정의
    title_style = ParagraphStyle('TitleStyle', fontName=font_name, fontSize=18, leading=22, spaceAfter=15)
    normal_style = ParagraphStyle('NormalStyle', fontName=font_name, fontSize=11, leading=16, spaceAfter=8)

    # 데이터 매핑
    if data.get('title'):
        story.append(Paragraph(f"<b>{data['title']}</b>", title_style))
    if data.get('author'):
        story.append(Paragraph(f"<b>작성자:</b> {data['author']}", normal_style))
    if data.get('date'):
        story.append(Paragraph(f"<b>작성일:</b> {data['date']}", normal_style))
    
    story.append(Spacer(1, 12))

    if data.get('summary'):
        story.append(Paragraph(f"<b>요약:</b> {data['summary']}", normal_style))
        story.append(Spacer(1, 12))

    if data.get('items') and isinstance(data['items'], list):
        story.append(Paragraph("<b>세부 항목:</b>", normal_style))
        for item in data['items']:
            story.append(Paragraph(f"• {item}", normal_style))

    # PDF 저장
    doc.build(story)

def process_batch(input_dir, output_dir):
    """지정된 폴더의 모든 YAML 파일을 PDF로 일괄 변환"""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # 출력 폴더가 없으면 생성
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 폰트 로드
    font_name = register_korean_font()

    # .yaml 및 .yml 확장자 파일 모두 검색
    yaml_files = list(in_path.glob('*.yaml')) + list(in_path.glob('*.yml'))
    
    if not yaml_files:
        print(f"'{input_dir}' 폴더에 처리할 YAML 파일이 없습니다.")
        return

    print(f"총 {len(yaml_files)}개의 파일을 찾았습니다. 변환을 시작합니다...\n")

    success_count = 0
    fail_count = 0

    for yaml_file in yaml_files:
        # 출력될 PDF 파일 경로 설정 (기존 파일명.pdf)
        pdf_filename = f"{yaml_file.stem}.pdf"
        pdf_filepath = out_path / pdf_filename

        try:
            # YAML 파싱
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # YAML 파일이 비어있거나 딕셔너리가 아닌 경우 처리
            if not isinstance(data, dict):
                raise ValueError("유효한 YAML 형식이 아닙니다 (Key-Value 구조 필요).")

            # PDF 생성
            create_pdf(data, pdf_filepath, font_name)
            print(f"[성공] {yaml_file.name} -> {pdf_filename}")
            success_count += 1

        except Exception as e:
            print(f"[실패] {yaml_file.name} 변환 중 오류 발생: {e}")
            fail_count += 1

    print("\n==================================")
    print(f"배치 처리 완료! (성공: {success_count}건, 실패: {fail_count}건)")
    print(f"결과물 저장 위치: {out_path.absolute()}")
    print("==================================")

if __name__ == "__main__":
    # 입력 폴더와 출력 폴더를 지정하여 실행
    INPUT_FOLDER = "input_yamls"
    OUTPUT_FOLDER = "output_pdfs"
    
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)