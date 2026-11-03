"""
무역 PDF → 페이지 PNG + 단어 위치
================================

앱은 PDF 를 이미지로 바꾸지 않습니다. 텍스트 PDF 는 텍스트 경로로만 가고, SigLIP2 경로는 이미지 파일을 받을 때만 돕니다.
그래서 비전 경로를 평가하려면 페이지 이미지를 따로 만들어야 합니다. batch_yaml_to_image_pdf.py 와 같은 200 DPI 입니다.

단어 위치(<stem>_p<n>.words.json)는 '값이 페이지 어디에 인쇄되어 있는가' 의 정답입니다.
히트맵과 크롭이 값을 덮는지 이 좌표로 잽니다.

  python rasterize_trading.py <출력 폴더>        (PyMuPDF 필요)
"""
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
DPI = 200


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    zoom = DPI / 72
    pages = 0
    for pdf in sorted((ROOT / "trading" / "output_pdfs").glob("*.pdf")):
        doc = fitz.open(pdf)
        for n, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pix.save(out / f"{pdf.stem}_p{n}.png")
            words = [
                {"x0": w[0] * zoom, "y0": w[1] * zoom, "x1": w[2] * zoom, "y1": w[3] * zoom, "text": w[4], "line": w[6]}
                for w in page.get_text("words")
            ]
            (out / f"{pdf.stem}_p{n}.words.json").write_text(
                json.dumps({"width": pix.width, "height": pix.height, "words": words}, ensure_ascii=False), encoding="utf-8"
            )
            pages += 1
        doc.close()
    print(f"{pages}페이지 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
