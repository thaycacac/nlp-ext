"""Chỉ số, biểu đồ, phân tích lỗi và đánh giá tra cứu fastText động."""
import csv
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C
from .data import encode_split
from .train import make_loader, predict


# --- Chỉ số -------------------------------------------------------------------
def compute_metrics(y_true, probs, threshold=0.5):
    """accuracy, precision, recall, F1 (lớp dương), macro-F1, ROC-AUC."""
    y_pred = (np.asarray(probs) > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
    }


def run_id(exp, emb, trainable, seed):
    return f"{exp}_{emb}_{'trainable' if trainable else 'frozen'}_seed{seed}"


def load_runs():
    """Đọc mọi run đã lưu: danh sách (meta+metrics, history)."""
    runs = []
    for p in sorted(C.RUNS_DIR.glob("*.json")):
        if p.name.endswith("_dynamic.json"):
            continue
        runs.append(json.load(open(p)))
    return runs


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
RARE_KEYS = ["rare_accuracy", "rare_macro_f1", "rare_roc_auc"]


def write_metrics_csv(rows):
    """results/metrics.csv: một dòng cho MỖI run (thí nghiệm x embedding x trainable x seed)."""
    cols = (["run_id", "experiment", "embedding", "trainable", "seed"] + METRIC_KEYS
            + RARE_KEYS + ["best_epoch", "epochs_run", "best_val_loss", "train_time_s",
                           "trainable_params", "total_params", "peak_rss_mb", "device"])
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(C.RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(rows):
    """results/summary.csv: trung bình ± độ lệch chuẩn theo (thí nghiệm, embedding, trainable)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["embedding"], r["trainable"]), []).append(r)
    cols = ["experiment", "embedding", "trainable", "n_seeds"]
    keys = METRIC_KEYS + RARE_KEYS + ["train_time_s", "trainable_params", "best_epoch"]
    for k in keys:
        cols += [f"{k}_mean", f"{k}_std"]
    out = []
    for (exp, emb, tr), rs in groups.items():
        row = {"experiment": exp, "embedding": emb, "trainable": tr, "n_seeds": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs if r.get(k) is not None], dtype=float)
            row[f"{k}_mean"] = float(v.mean()) if len(v) else ""
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    with open(C.RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(out, key=lambda r: (r["trainable"], r["experiment"], r["embedding"])):
            w.writerow(r)
    return out


# --- Biểu đồ ------------------------------------------------------------------
LABELS = {"random": "Random", "word2vec": "Word2Vec CBOW", "glove": "GloVe",
          "fasttext": "fastText CBOW"}


def _runs_for(runs, trainable):
    conds = {}
    for r in runs:
        if r["trainable"] == trainable and r["experiment"] != "E3dyn":
            conds.setdefault((r["experiment"], r["embedding"]), []).append(r)
    return conds


def plot_learning_curves(runs, trainable, path):
    """Loss train (nét liền) và validation (nét đứt) theo epoch, mỗi seed một màu."""
    conds = _runs_for(runs, trainable)
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ((exp, emb), rs) in zip(axes, sorted(conds.items())):
        for r in sorted(rs, key=lambda r: r["seed"]):
            ep = [h["epoch"] for h in r["history"]]
            c = f"C{r['seed']}"
            ax.plot(ep, [h["train_loss"] for h in r["history"]], "-", color=c,
                    label=f"train s{r['seed']}")
            ax.plot(ep, [h["val_loss"] for h in r["history"]], "--", color=c,
                    label=f"val s{r['seed']}")
        ax.set_title(f"{exp}: {LABELS[emb]}")
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BCE loss")
    axes[-1].legend(fontsize=7, ncol=2)
    kind = "trainable" if trainable else "frozen"
    fig.suptitle(f"Learning curves ({kind} embedding; solid=train, dashed=validation)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion_matrices(runs, y_test, trainable, path):
    """Confusion matrix (seed nhỏ nhất của mỗi điều kiện) trên tập test chính thức."""
    conds = _runs_for(runs, trainable)
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(3.8 * len(conds), 3.6))
    axes = np.atleast_1d(axes)
    for ax, ((exp, emb), rs) in zip(axes, sorted(conds.items())):
        r = min(rs, key=lambda r: r["seed"])
        probs = np.load(C.RUNS_DIR / f"{r['run_id']}_probs.npy")
        cm = confusion_matrix(y_test, probs > 0.5)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{exp}: {LABELS[emb]} (seed {r['seed']})\nacc={r['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Phân tích lỗi ------------------------------------------------------------
_NEG = {"not", "no", "never", "nothing", "neither", "nor", "hardly", "barely", "without"}
_CONTRAST = {"but", "however", "although", "though", "despite", "yet", "except"}


def tag_review(tokens, unk_rate, mean_prob):
    """Gán nhãn chủ đề lỗi bằng heuristic đơn giản (sarcasm cần đọc tay)."""
    tags = []
    n_neg = sum(t in _NEG or t.endswith("n't") for t in tokens)
    if n_neg >= 3:
        tags.append("negation")
    if sum(t in _CONTRAST for t in tokens) >= 3:
        tags.append("contrast/mixed")
    if len(tokens) > C.MAX_LEN:
        tags.append(f"long(>{C.MAX_LEN}: cuối bị cắt)")
    if unk_rate > 0.05:
        tags.append("rare-words")
    if abs(mean_prob - 0.5) < 0.15:
        tags.append("ambiguous")
    return tags or ["(none)"]


def error_analysis(test, rare_rates, runs, n_hard=12, n_random=8):
    """Tìm review test mà TẤT CẢ bốn embedding frozen (E0-E3) đều đoán sai.

    Mỗi điều kiện dùng xác suất trung bình qua các seed. Lấy n_hard review sai
    "tự tin" nhất + n_random review sai khác chọn ngẫu nhiên; ghi ra markdown.
    """
    y = test["labels"]
    cond_probs = {}
    for r in runs:
        if not r["trainable"] and r["experiment"] in ("E0", "E1", "E2", "E3"):
            cond_probs.setdefault(r["embedding"], []).append(
                np.load(C.RUNS_DIR / f"{r['run_id']}_probs.npy"))
    if len(cond_probs) < 4:
        return None
    mean = {k: np.mean(v, axis=0) for k, v in cond_probs.items()}
    wrong = np.stack([(p > 0.5).astype(int) != y for p in mean.values()])   # [4, N]
    all_wrong = np.where(wrong.all(axis=0))[0]
    avg_prob = np.mean(list(mean.values()), axis=0)
    # độ "tự tin sai" = khoảng cách từ xác suất TB đến nhãn thật
    conf = np.abs(avg_prob - y)
    order = all_wrong[np.argsort(-conf[all_wrong])]
    hard = list(order[:n_hard])
    rest = [i for i in order[n_hard:]]
    rs = np.random.RandomState(0)
    rand = list(rs.choice(rest, size=min(n_random, len(rest)), replace=False)) if rest else []
    lines = [f"# Error analysis (review test bị TẤT CẢ E0-E3 đoán sai)\n",
             f"Số review bị cả 4 điều kiện đoán sai: **{len(all_wrong)} / {len(y)}** "
             f"({100 * len(all_wrong) / len(y):.1f}%).\n",
             f"Dưới đây: {len(hard)} review sai tự tin nhất + {len(rand)} review sai ngẫu nhiên. "
             "Nhãn chủ đề tự động chỉ là heuristic; chủ đề cuối cùng xem REPORT.md.\n"]
    tag_count = {}
    for rank, i in enumerate(hard + rand, 1):
        toks = test["tokens"][i]
        tags = tag_review(toks, rare_rates[i], avg_prob[i])
        for t in tags:
            tag_count[t] = tag_count.get(t, 0) + 1
        text = re.sub(r"<br\s*/?>", " ", test["texts"][i]).strip()
        snippet = text if len(text) <= 700 else text[:450] + " [...] " + text[-250:]
        lines += [f"## {rank}. test[{i}] — nhãn thật: {'POS' if y[i] else 'NEG'}; "
                  f"xác suất dương TB: {avg_prob[i]:.2f}; {len(toks)} token; "
                  f"unk={100 * rare_rates[i]:.1f}%",
                  f"Tags: {', '.join(tags)}\n", f"> {snippet}\n"]
    lines.append("## Tổng hợp nhãn heuristic\n")
    lines += [f"- {k}: {v}" for k, v in sorted(tag_count.items(), key=lambda kv: -kv[1])]
    (C.RESULTS_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return {"all_wrong": int(len(all_wrong)), "n": int(len(y)), "tags": tag_count}


# --- Tra cứu fastText động (extension) -----------------------------------------
def dynamic_fasttext_eval(model, ft, vocab, test, rare_idx, device):
    """Suy luận với vector fastText cho TỪ NGOÀI vocabulary thay vì <unk>.

    model: LSTM đã huấn luyện với embedding fastText frozen (giữ nguyên trọng số).
    Với mỗi token của test không có trong vocabulary huấn luyện, lấy
    ft.wv.get_vector(token) (tổng các n-gram ký tự), thêm vào bảng embedding
    như hàng mới rồi mã hóa lại test với vocabulary mở rộng. Không dùng nhãn hay
    thống kê test để huấn luyện; chỉ tra vector ở thời điểm suy luận (giống khi triển khai).
    """
    new_tokens = sorted({t for toks in test["tokens"] for t in toks if t not in vocab})
    ext_vocab = dict(vocab)
    for t in new_tokens:
        ext_vocab[t] = len(ext_vocab)
    extra = np.stack([ft.wv.get_vector(t) for t in new_tokens]) if new_tokens else \
        np.zeros((0, C.EMB_DIM), dtype=np.float32)
    model.extend_embedding(extra)
    ids, lengths = encode_split(test["tokens"], ext_vocab)
    loader = make_loader(ids, lengths, test["labels"], shuffle=False)
    _, probs, labels = predict(model, loader, device)
    return probs, labels, len(new_tokens)
