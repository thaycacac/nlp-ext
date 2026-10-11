"""Chạy toàn bộ thí nghiệm của assignment: E0-E4 x seeds + đánh giá + báo cáo số liệu.

Cách chạy (từ thư mục 04-assignment, dùng .venv riêng của project này):
    .venv/bin/python run_experiments.py                  # đầy đủ (3 seed)
    .venv/bin/python run_experiments.py --seeds 0        # pilot 1 seed
    .venv/bin/python run_experiments.py --quick          # chạy thử rất nhanh (dữ liệu nhỏ)
Chương trình có thể ngắt rồi chạy lại: run nào đã có kết quả trong results/runs/ sẽ được bỏ qua.
"""
import argparse
import csv
import json
import resource
import sys
import time

import numpy as np
import torch

from src import config as C
from src import data as D
from src import embeddings as E
from src import evaluate as V
from src.model import SentimentLSTM, count_parameters
from src.train import get_device, make_loader, predict, set_seed, train_model


def peak_rss_mb():
    """RSS đỉnh của TIẾN TRÌNH (tích lũy, không reset giữa các run). macOS tính bằng byte."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="dữ liệu nhỏ, 2 epoch (smoke test)")
    ap.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--only", nargs="+", help="chỉ chạy các thí nghiệm này, vd E1 E3")
    args = ap.parse_args()

    if args.quick:   # smoke test ghi ra thư mục riêng để không lẫn với kết quả thật
        C.RESULTS_DIR = C.ROOT / "results_quick"
        C.RUNS_DIR = C.RESULTS_DIR / "runs"
        C.CACHE_DIR = C.ROOT / "cache_quick"
        for m in (D, E, V):
            m.C = C
        args.max_epochs = 2
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device}")

    # ---- 1. Dữ liệu, vocabulary (chỉ từ TRAIN) ---------------------------------
    print("\n[1] Nạp IMDB + tokenize + vocabulary")
    data = D.load_imdb(limit=2000 if args.quick else None)
    vocab = D.build_vocab(data["train"]["tokens"])
    stats = {s: D.unk_stats(data[s]["tokens"], vocab) for s in ("train", "val", "test")}
    print(f"  train/val/test = {[len(data[s]['labels']) for s in ('train','val','test')]}, "
          f"vocab = {len(vocab)} (gồm <pad>, <unk>)")
    for s, st in stats.items():
        print(f"  {s:5s}: {st['tokens']:>9,} token | <unk> token {st['unk_token_pct']:.2f}% | "
              f"<unk> type {st['unk_type_pct']:.2f}%")
    enc = {s: D.encode_split(data[s]["tokens"], vocab) for s in ("train", "val", "test")}

    rare_rates = D.review_unk_rates(data["test"]["tokens"], vocab)
    n_rare = max(1, len(rare_rates) // 10)
    rare_idx = np.argsort(-rare_rates)[:n_rare]       # 10% review test nhiều <unk> nhất
    json.dump(rare_idx.tolist(), open(C.RESULTS_DIR / "rare_subset_indices.json", "w"))
    print(f"  Tập con từ hiếm: {n_rare} review test có tỉ lệ <unk> cao nhất "
          f"(>= {rare_rates[rare_idx].min() * 100:.1f}%)")

    # ---- 2. Embedding ----------------------------------------------------------
    print("\n[2] Embedding: Word2Vec CBOW, fastText CBOW (huấn luyện trên TRAIN), GloVe (công khai)")
    w2v, w2v_info = E.train_word2vec(data["train"]["tokens"])
    ft, ft_info = E.train_fasttext(data["train"]["tokens"])
    glove, glove_info = E.load_glove()
    print(f"  word2vec {w2v_info['train_time_s']:.0f}s, fasttext {ft_info['train_time_s']:.0f}s, "
          f"glove nạp {glove_info['load_time_s']:.0f}s")

    sources = {
        "word2vec": E.kv_lookup(w2v.wv),
        "glove": E.kv_lookup(glove),
        "fasttext": lambda t: ft.wv[t] if t in ft.wv.key_to_index else None,
    }
    matrices, founds, coverage_rows = {}, {}, []
    for name, look in sources.items():
        m, found, st = E.build_embedding_matrix(vocab, look, seed=C.EMB_SEED)
        matrices[name], founds[name] = m, found
        row = {"source": name, "vocab_words": st["vocab_words"], "found": st["found"],
               "unique_coverage_pct": round(st["unique_coverage_pct"], 2)}
        for s in ("train", "val", "test"):
            row[f"token_coverage_{s}_pct"] = round(
                E.token_coverage(data[s]["tokens"], vocab, found), 2)
        row["unk_val_pct"] = round(stats["val"]["unk_token_pct"], 2)
        row["unk_test_pct"] = round(stats["test"]["unk_token_pct"], 2)
        row["train_time_s"] = round({"word2vec": w2v_info, "glove": glove_info,
                                     "fasttext": ft_info}[name]["train_time_s"], 1)
        coverage_rows.append(row)
        print(f"  {name:9s} coverage: từ vựng {row['unique_coverage_pct']:.1f}% | "
              f"token test {row['token_coverage_test_pct']:.1f}%")
    with open(C.RESULTS_DIR / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0]))
        w.writeheader()
        w.writerows(coverage_rows)
    (C.RESULTS_DIR / "subword_demo.txt").write_text(E.subword_demo(w2v, ft), encoding="utf-8")
    print("  " + E.subword_demo(w2v, ft).replace("\n", "\n  "))

    # ---- 3. Các run: E0-E4 x seeds ---------------------------------------------
    print("\n[3] Huấn luyện LSTM")
    y_test = data["test"]["labels"]
    val_loader = make_loader(*enc["val"], data["val"]["labels"], shuffle=False)
    test_loader = make_loader(*enc["test"], y_test, shuffle=False)
    exps = [e for e in C.EXPERIMENTS if not args.only or e[0] in args.only]
    for seed in args.seeds:
        for exp, emb, trainable in exps:
            rid = V.run_id(exp, emb, trainable, seed)
            if (C.RUNS_DIR / f"{rid}.json").exists():
                print(f"  {rid}: đã có, bỏ qua")
                continue
            print(f"  {rid}")
            set_seed(seed)
            matrix = E.random_matrix(vocab, seed=seed) if emb == "random" else matrices[emb]
            model = SentimentLSTM(matrix, hidden_dim=C.HIDDEN_DIM, trainable=trainable)
            train_loader = make_loader(*enc["train"], data["train"]["labels"],
                                       shuffle=True, seed=seed)
            history, best_epoch, secs = train_model(
                model, train_loader, val_loader, device, max_epochs=args.max_epochs)
            _, probs, _ = predict(model, test_loader, device)
            m = V.compute_metrics(y_test, probs)
            rm = V.compute_metrics(y_test[rare_idx], probs[rare_idx])
            tp, total = count_parameters(model)
            rec = {"run_id": rid, "experiment": exp, "embedding": emb, "trainable": trainable,
                   "seed": seed, **m,
                   "rare_accuracy": rm["accuracy"], "rare_macro_f1": rm["macro_f1"],
                   "rare_roc_auc": rm["roc_auc"], "best_epoch": best_epoch,
                   "epochs_run": len(history),
                   "best_val_loss": min(h["val_loss"] for h in history),
                   "train_time_s": secs, "trainable_params": tp, "total_params": total,
                   "peak_rss_mb": peak_rss_mb(), "device": str(device), "history": history}
            np.save(C.RUNS_DIR / f"{rid}_probs.npy", probs)
            json.dump(rec, open(C.RUNS_DIR / f"{rid}.json", "w"))
            if exp == "E3" and not trainable:   # giữ checkpoint để tra cứu fastText động
                (C.CACHE_DIR / "ckpt").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), C.CACHE_DIR / "ckpt" / f"{rid}.pt")
            print(f"    -> test acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
                  f"AUC {m['roc_auc']:.4f} | {secs:.0f}s", flush=True)
            V.write_metrics_csv(V.load_runs())

    # ---- 4. Tra cứu fastText động cho từ chưa thấy (extension) ------------------
    print("\n[4] fastText động (get_vector) so với <unk>")
    for seed in args.seeds:
        base = V.run_id("E3", "fasttext", False, seed)
        ckpt = C.CACHE_DIR / "ckpt" / f"{base}.pt"
        rid = V.run_id("E3dyn", "fasttext", False, seed)
        if not ckpt.exists() or (C.RUNS_DIR / f"{rid}.json").exists():
            continue
        model = SentimentLSTM(matrices["fasttext"], hidden_dim=C.HIDDEN_DIM, trainable=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        probs, _, n_new = V.dynamic_fasttext_eval(model, ft, vocab, data["test"], rare_idx, device)
        m = V.compute_metrics(y_test, probs)
        rm = V.compute_metrics(y_test[rare_idx], probs[rare_idx])
        rec = {"run_id": rid, "experiment": "E3dyn", "embedding": "fasttext", "trainable": False,
               "seed": seed, **m, "rare_accuracy": rm["accuracy"],
               "rare_macro_f1": rm["macro_f1"], "rare_roc_auc": rm["roc_auc"],
               "new_tokens": n_new, "history": [], "device": str(device)}
        np.save(C.RUNS_DIR / f"{rid}_probs.npy", probs)
        json.dump(rec, open(C.RUNS_DIR / f"{rid}.json", "w"))
        print(f"  seed {seed}: {n_new} từ mới | acc {m['accuracy']:.4f} | rare macro-F1 "
              f"{rm['macro_f1']:.4f}")

    # ---- 5. Tổng hợp, biểu đồ, phân tích lỗi ------------------------------------
    print("\n[5] Tổng hợp kết quả")
    runs = V.load_runs()
    V.write_metrics_csv(runs)
    V.write_summary_csv(runs)
    for trainable, suffix in ((False, ""), (True, "_trainable")):
        V.plot_learning_curves(runs, trainable, C.RESULTS_DIR / f"learning_curves{suffix}.png")
        V.plot_confusion_matrices(runs, y_test, trainable,
                                  C.RESULTS_DIR / f"confusion_matrices{suffix}.png")
    ea = V.error_analysis(data["test"], rare_rates, runs)
    print(f"  Error analysis: {ea}")
    print(f"\nKết quả nằm ở {C.RESULTS_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Tổng thời gian: {(time.time() - t0) / 60:.1f} phút")
