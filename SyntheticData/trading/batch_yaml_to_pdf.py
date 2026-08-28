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

def parse_yaml_to_story(data, story, style, level=0):
    """
    YAML 데이터를 재귀적으로 탐색하며 PDF 스토리(Paragraph)에 추가하는 함수
    - level 변수로 중첩 깊이에 따른 들여쓰기를 구현
    """
    # HTML 엔티티를 사용하여 들여쓰기 구현 (1 level 당 스페이스 4칸)
    indent = "&nbsp;" * (level * 4) 
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # 값이 다시 딕셔너리나 리스트면 키만 먼저 출력하고 한 단계 깊게 들어감
                story.append(Paragraph(f"{indent}<b>{key}</b>:", style))
                parse_yaml_to_story(value, story, style, level + 1)
            else:
                # 일반 값이면 '키: 값' 형태로 출력
                story.append(Paragraph(f"{indent}<b>{key}</b>: {value}", style))
                
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                # 리스트 내부가 딕셔너리일 경우 인덱스로 항목 구분
                story.append(Paragraph(f"{indent}<b>[ Item {idx + 1} ]</b>", style))
                parse_yaml_to_story(item, story, style, level + 1)
            elif isinstance(item, list):
                parse_yaml_to_story(item, story, style, level + 1)
            else:
                # 리스트 내부가 단순 텍스트일 경우 불릿 기호 추가
                story.append(Paragraph(f"{indent}• {item}", style))
    else:
         story.append(Paragraph(f"{indent}{data}", style))

def create_pdf(data, pdf_path, font_name):
    """파싱된 데이터를 PDF로 렌더링"""
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []

    # 전체 본문에 적용될 폰트 스타일
    normal_style = ParagraphStyle(
        'NormalStyle', 
        fontName=font_name, 
        fontSize=10, 
        leading=16, 
        spaceAfter=6
    )
    
    # 재귀 파서 호출 시작
    parse_yaml_to_story(data, story, normal_style)

    # PDF 저장
    doc.build(story)

def process_batch(input_dir, output_dir):
    """지정된 폴더의 모든 YAML 파일을 PDF로 일괄 변환"""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    font_name = register_korean_font()

    yaml_files = list(in_path.glob('*.yaml')) + list(in_path.glob('*.yml'))
    
    if not yaml_files:
        print(f"'{input_dir}' 폴더에 처리할 YAML 파일이 없습니다.")
        return

    print(f"총 {len(yaml_files)}개의 파일을 찾았습니다. 변환을 시작합니다...\n")

    success_count = 0
    fail_count = 0

    for yaml_file in yaml_files:
        pdf_filename = f"{yaml_file.stem}.pdf"
        pdf_filepath = out_path / pdf_filename

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                print(f"[스킵] {yaml_file.name} (파일이 비어있습니다.)")
                continue

            create_pdf(data, pdf_filepath, font_name)
            print(f"[성공] {yaml_file.name} -> {pdf_filename}")
            success_count += 1

        except Exception as e:
            print(f"[실패] {yaml_file.name} 변환 중 오류 발생: {e}")
            fail_count += 1

    print("\n==================================")
    print(f"배치 처리 완료! (성공: {success_count}건, 실패: {fail_count}건)")
    print("==================================")

if __name__ == "__main__":
    INPUT_FOLDER = "input_yamls"
    OUTPUT_FOLDER = "output_pdfs"
    
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)