"""
무역 PDF → 페이지 PNG + 단어 위치
================================

앱은 PDF 를 이미지로 바꾸지 않습니다. 텍스트 PDF 는 텍스트 경로로만 가고, SigLIP2 경로는 이미지 파일을 받을 때만 돕니다.
그래서 비전 경로를 평가하려면 페이지 이미지를 따로 만들어야 합니다. batch_yaml_to_image_pdf.py 와 같은 200 DPI 입니다.

단어 위치(<stem>_p<n>.words.json)는 '값이 페이지 어디에 인쇄되어 있는가' 의 정답입니다.
히트맵과 크롭이 값을 덮는지 이 좌표로 잽니다.

  python rasterize_trading.py <출력 폴더> [PDF 폴더]

렌더러는 pypdfium2(PDFium 바인딩, BSD-3-Clause / Apache-2.0)입니다. 예전에는 PyMuPDF(AGPL-3.0)를 썼습니다.
PDFium 은 글자 상자만 주므로 공백 · 줄바꿈에서 끊어 낱말로 묶습니다 (PyMuPDF 의 get_text("words") 와 같은 단위).
"""
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
DPI = 200


def page_words(page, zoom: float) -> list:
    """페이지 낱말 상자(픽셀, 위쪽 원점). PDF 좌표는 왼쪽 아래 원점이라 y 를 뒤집습니다."""
    textpage = page.get_textpage()
    height = page.get_height()
    words, cur, line = [], None, 0
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if not ch or ch.isspace():
            if cur:
                words.append(cur)
                cur = None
            if ch == "\n":
                line += 1
            continue
        left, bottom, right, top = textpage.get_charbox(i)
        x0, y0, x1, y1 = left * zoom, (height - top) * zoom, right * zoom, (height - bottom) * zoom
        if cur is None:
            cur = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": ch, "line": line}
        else:
            cur["x0"], cur["y0"] = min(cur["x0"], x0), min(cur["y0"], y0)
            cur["x1"], cur["y1"] = max(cur["x1"], x1), max(cur["y1"], y1)
            cur["text"] += ch
    if cur:
        words.append(cur)
    textpage.close()
    return words


def main() -> int:
    out = Path(sys.argv[1])
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "trading" / "output_pdfs"
    out.mkdir(parents=True, exist_ok=True)
    zoom = DPI / 72
    pages = 0
    for path in sorted(src.glob("*.pdf")):
        pdf = pdfium.PdfDocument(str(path))
        for n in range(len(pdf)):
            page = pdf[n]
            image = page.render(scale=zoom).to_pil().convert("RGB")
            image.save(out / f"{path.stem}_p{n + 1}.png")
            words = page_words(page, zoom)
            (out / f"{path.stem}_p{n + 1}.words.json").write_text(
                json.dumps({"width": image.width, "height": image.height, "words": words}, ensure_ascii=False), encoding="utf-8"
            )
            page.close()
            pages += 1
        pdf.close()
    print(f"{pages}페이지 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
