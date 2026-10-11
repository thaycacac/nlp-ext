"""Sinh báo cáo PDF "8 Required experiments" từ kết quả trong results*/.

Dùng:
    .venv/bin/python make_report.py --results results_quick --out report/report_quick.pdf   # tạm thời (quick)
    .venv/bin/python make_report.py --results results       --out report/report.pdf         # kết quả thật
Cần Google Chrome (in HTML -> PDF bằng chế độ headless); không cần thêm thư viện Python.
"""
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ROWS = [  # (run, điều kiện, layer, mục đích, [(nhãn, experiment, embedding, trainable)])
    ("E0", "Random 100-dimensional vectors", "Frozen or fixed control",
     "Measure what the LSTM learns without external semantic initialization.",
     [("", "E0", "random", False)]),
    ("E1", "Word2Vec CBOW trained on IMDB training text", "Frozen",
     "Test word-level local context representations.", [("", "E1", "word2vec", False)]),
    ("E2", "GloVe 100-dimensional public vectors", "Frozen",
     "Test global co-occurrence representations from an external corpus.",
     [("", "E2", "glove", False)]),
    ("E3", "fastText CBOW trained on IMDB training text", "Frozen",
     "Test the value of character n-grams and subword information.",
     [("", "E3", "fasttext", False)]),
    ("E4", "Word2Vec, GloVe, and fastText repeated", "Trainable",
     "Measure how much task-specific fine-tuning changes the comparison.",
     [("Word2Vec", "E4", "word2vec", True), ("GloVe", "E4", "glove", True),
      ("fastText", "E4", "fasttext", True)]),
]

CSS = """
@page { size: A4 landscape; margin: 14mm 12mm; }
body { font-family: 'Times New Roman', Times, serif; color: #111; margin: 0; }
h1 { font-size: 22px; margin: 0 0 8px; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 11.5px; }
tr { break-inside: avoid; }
th { background: #2f5077; color: #fff; text-align: left; padding: 5px 8px;
     border: 1px solid #c9d1dc; font-size: 12px; }
td { border: 1px solid #c9d1dc; padding: 4px 8px; vertical-align: middle; line-height: 1.25; }
tr.done td { background: #e9f0f7; }
td.run { font-weight: bold; text-align: center; font-size: 13px; }
td.mid { text-align: center; }
.sub { display: flex; gap: 14px; }
.sub > div { flex: 1; }
b.n { font-weight: bold; }
.notes { margin-top: 12px; font-size: 12px; line-height: 1.5; }
.warn { margin-top: 8px; padding: 6px 9px; border: 1px solid #d9a441; background: #fdf4e0;
        font-size: 12px; }
"""


def ms(values, pct=False, digits=4, bold=False):
    """'mean ± std' (std theo ddof=1; bỏ khi chỉ có 1 seed)."""
    v = np.array(values, dtype=float) * (100 if pct else 1)
    d = 2 if pct else digits
    mean = f"{v.mean():.{d}f}" + ("%" if pct else "")
    txt = mean + (f" ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else "")
    return f"<b class='n'>{txt}</b>" if bold else txt


def result_block(rs, title=""):
    if title:   # dạng gọn cho ô E4 (3 cột): mỗi chỉ số một dòng, 3 chữ số thập phân
        g = lambda k: [r[k] for r in rs]
        return (f"<b>{title}</b><br>Accuracy: {ms(g('accuracy'), pct=True, digits=3, bold=True)}<br>"
                f"Precision: {ms(g('precision'), digits=3)}<br>Recall: {ms(g('recall'), digits=3)}<br>"
                f"F1 (positive): {ms(g('f1'), digits=3)}<br>"
                f"Macro-F1: {ms(g('macro_f1'), digits=3, bold=True)}<br>"
                f"ROC-AUC: {ms(g('roc_auc'), digits=3, bold=True)}")
    seeds = ", ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
    g = lambda k: [r[k] for r in rs]
    head = f"<b>{title}</b><br>" if title else ""
    return (f"<b>Completed - seed{'s' if len(rs) > 1 else ''} {seeds}</b>"
            f"{' (mean ± std)' if len(rs) > 1 else ''}<br>" if not title else head) + (
        f"Accuracy: {ms(g('accuracy'), pct=True, bold=True)}<br>"
        f"Precision / Recall: {ms(g('precision'))} / {ms(g('recall'))}<br>"
        f"F1 (positive): {ms(g('f1'))}<br>"
        f"Macro-F1: {ms(g('macro_f1'), bold=True)}<br>"
        f"ROC-AUC: {ms(g('roc_auc'), bold=True)}")


def best_config(groups, layer):
    """Ô "Best configuration": cấu hình LSTM cố định + epoch tốt nhất (chọn theo validation loss)
    của từng seed; với nhiều embedding (E4) thì nêu embedding có validation loss thấp nhất.
    Chọn theo VALIDATION, không dùng test, để tránh rò rỉ."""
    lines = []
    for label, rs in groups:
        rs = sorted(rs, key=lambda r: r["seed"])
        ep = " / ".join(str(r["best_epoch"]) for r in rs)
        vl = float(np.mean([r["best_val_loss"] for r in rs]))
        prefix = f"<b>{label}:</b> " if label else ""
        lines.append(f"{prefix}best epoch {ep} (seeds {'/'.join(str(r['seed']) for r in rs)}); "
                     f"mean val loss {vl:.3f}")
    if len(groups) > 1:
        scores = {label: np.mean([r["best_val_loss"] for r in rs]) for label, rs in groups}
        top = min(scores, key=scores.get)
        lines.append(f"<b>Best (lowest val loss): {top}</b>")
    base = (f"{layer} embedding; LSTM 1x128, Adam 1e-3, batch 64, patience 2" if len(groups) == 1
            else "Trainable embedding; LSTM 1x128, Adam 1e-3, batch 64, patience 2")
    return base + "<br>" + "<br>".join(lines)


def build(results_dir: Path, provisional: bool):
    runs = [json.load(open(p)) for p in sorted((results_dir / "runs").glob("*.json"))
            if not p.name.endswith("_dynamic.json")]
    sel = lambda exp, emb, tr: [r for r in runs if r["experiment"] == exp
                                and r["embedding"] == emb and r["trainable"] == tr]
    all_seeds = sorted({r["seed"] for r in runs if r["experiment"] != "E3dyn"})
    n_test = len(np.load(next((results_dir / "runs").glob("*_probs.npy"))))
    epochs_run = [r["epochs_run"] for r in runs if r.get("epochs_run")]
    best_epochs = [r["best_epoch"] for r in runs if r.get("best_epoch")]

    body = []
    for run, cond, layer, purpose, parts in ROWS:
        groups = [(label, sel(e, m, t)) for label, e, m, t in parts]
        done = all(rs for _, rs in groups)
        if not done:
            cell = "Not performed"
        elif len(groups) == 1:
            cell = result_block(groups[0][1])
        else:
            seeds = ", ".join(map(str, sorted({r["seed"] for _, rs in groups for r in rs})))
            cell = (f"<b>Completed - seeds {seeds} (mean ± std)</b><div class='sub'>"
                    + "".join(f"<div>{result_block(rs, label)}</div>" for label, rs in groups)
                    + "</div>")
        best = best_config(groups, layer.split()[0]) if done else "Not performed"
        body.append(f"<tr class='{'done' if done else ''}'><td class='run'>{run}</td>"
                    f"<td class='mid'>{html.escape(cond)}</td><td class='mid'>{layer}</td>"
                    f"<td>{html.escape(purpose)}</td><td>{best}</td><td>{cell}</td></tr>")

    if provisional:
        evaluation = (f"<b>Evaluation:</b> PROVISIONAL quick-mode pilot. Test set = random sample of "
                      f"{n_test:,} reviews from the official IMDB test split (not the full 25,000); "
                      f"training used 2,000 reviews (500 validation) and at most "
                      f"{max(epochs_run)} epochs; one-layer LSTM (128 hidden units).")
    else:
        evaluation = (f"<b>Evaluation:</b> Official IMDB test set, {n_test:,} reviews; "
                      f"one-layer unidirectional LSTM (128 hidden units).")
    notes = (f"<div class='notes'>{evaluation}<br>"
             f"<b>Selection:</b> Best epoch chosen by validation loss (best epochs observed: "
             f"{min(best_epochs)}-{max(best_epochs)}); early stopping patience 2. "
             f"Seeds: {', '.join(map(str, all_seeds))}.<br>"
             f"<b>Source:</b> {results_dir.name}/metrics.csv (per run) and {results_dir.name}/summary.csv. "
             f"Precision, recall and positive-class F1 use label 1; threshold = 0.5.</div>")
    warn = ("<div class='warn'><b>Note:</b> these numbers come from the quick smoke test "
            "(tiny training set, 2 epochs) and are NOT final; they will be replaced by the "
            "full-run results (25,000 test reviews).</div>") if provisional else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Required experiments</title>"
            f"<style>{CSS}</style></head><body><h1>8 Required experiments</h1>"
            "<table><colgroup><col style='width:4%'><col style='width:12%'><col style='width:8%'>"
            "<col style='width:15%'><col style='width:19%'><col style='width:42%'></colgroup>"
            "<tr><th>Run</th><th>Embedding condition</th><th>Embedding layer</th><th>Purpose</th>"
            f"<th>Best configuration</th><th>Results</th></tr>{''.join(body)}</table>{notes}{warn}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/report.pdf")
    ap.add_argument("--provisional", action="store_true",
                    help="gắn nhãn tạm thời (tự bật nếu tên thư mục chứa 'quick')")
    a = ap.parse_args()
    results = (ROOT / a.results).resolve()
    out = (ROOT / a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    page = out.with_suffix(".html")
    page.write_text(build(results, a.provisional or "quick" in results.name), encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", page.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Đã tạo {out}")


if __name__ == "__main__":
    sys.exit(main())
