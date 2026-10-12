"""Xuất REPORT.md (kèm bảng thí nghiệm và các biểu đồ) thành PDF bằng Chrome headless.

    .venv/bin/python export_report_pdf.py            # -> report/technical_report.pdf
"""
import re
import subprocess
from pathlib import Path

import markdown

import make_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report" / "technical_report.pdf"

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5px; line-height: 1.42; color: #111; }
h1 { font-size: 21px; margin: 0 0 8px; } h2 { font-size: 16px; margin: 16px 0 6px; color: #2f5077; }
h3 { font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 10.5px; }
th { background: #2f5077; color: #fff; padding: 4px 6px; text-align: left; border: 1px solid #c9d1dc; }
td { border: 1px solid #c9d1dc; padding: 3px 6px; }
tr { break-inside: avoid; } img { width: 100%; break-inside: avoid; }
figure { margin: 10px 0; break-inside: avoid; } figcaption { font-size: 10.5px; color: #333; }
code { font-size: 10.5px; background: #f1f3f6; padding: 0 2px; }
"""

FIGURES = [
    ("results/learning_curves.png", "Figure 1. Training (solid) and validation (dashed) loss, frozen embeddings, three seeds."),
    ("results/learning_curves_trainable.png", "Figure 2. Training and validation loss, trainable embeddings: validation loss turns upward after epoch 3 to 6 while training loss keeps falling."),
    ("results/confusion_matrices.png", "Figure 3. Confusion matrices on the official test set, frozen embeddings (seed 0)."),
    ("results/confusion_matrices_trainable.png", "Figure 4. Confusion matrices on the official test set, trainable embeddings (seed 0)."),
]


def main():
    md = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    md = re.sub(r"([^\n])\n(- )", r"\1\n\n\2", md)   # python-markdown cần dòng trống trước danh sách
    body = markdown.markdown(md, extensions=["tables"])
    # đặt các hình ngay trước mục 3 (Answers)
    figs = "".join(f"<figure><img src='{(ROOT / p).as_uri()}'><figcaption>{c}</figcaption></figure>"
                   for p, c in FIGURES)
    body = body.replace("<h2>3. Answers", f"<h2>Figures</h2>{figs}<h2>3. Answers", 1)
    # bảng "Required experiments" (từ make_report) làm trang đầu
    table_html = make_report.build(ROOT / "results", provisional=False)
    table_body = re.search(r"<body>(.*)</body>", table_html, re.S).group(1)
    table_css = re.search(r"<style>(.*?)</style>", table_html, re.S).group(1)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>{CSS}"
            f"</style></head><body>{body}"
            f"<div style='break-before:page'><style>{table_css.replace('@page','@pagex')}"
            f"</style>{table_body}</div></body></html>")
    html_path = OUT.with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    subprocess.run([make_report.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT}", html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html_path.unlink()
    print(f"Đã tạo {OUT}")


if __name__ == "__main__":
    main()
