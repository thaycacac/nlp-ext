"""Nạp IMDB, tách train/validation, tokenize, xây vocabulary, pad/cắt chuỗi.

Quy tắc chống rò rỉ (leakage): vocabulary chỉ xây từ review của tập TRAIN
(80% của 25.000 review train chính thức). Tập validation và test chính thức
chỉ được ENCODE bằng vocabulary đó, không bao giờ tham gia thống kê.
"""
import json
import re
import ssl
from collections import Counter

import numpy as np

from . import config as C


def fix_ssl():
    """Python bản python.org trên macOS thiếu chứng chỉ gốc -> tải dữ liệu bị lỗi
    CERTIFICATE_VERIFY_FAILED. Trỏ SSL sang bộ chứng chỉ của certifi."""
    try:
        import certifi
        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
            cafile=certifi.where())
    except ImportError:
        pass


# --- Tokenizer duy nhất, dùng cho cả ba điều kiện embedding -------------------
# Chữ thường; chỉ giữ chữ cái a-z và dấu nháy ' (để giữ "don't", "it's").
# Dữ liệu IMDB chứa thẻ HTML "<br />" nên phải bỏ trước, nếu không "br" sẽ
# thành một token rất phổ biến.
_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str):
    text = text.lower().replace("<br />", " ")
    return [t for t in _TOKEN_RE.findall(text) if t.strip("'")]


def load_imdb(limit=None):
    """Trả về dict gồm texts/labels/tokens cho 3 split: train, val, test.

    - Tách 80/20 từ train chính thức bằng hoán vị ngẫu nhiên với SPLIT_SEED
      cố định; chỉ số lưu ra results/split_indices.json để tái lập.
    - `limit` (chỉ để chạy thử nhanh) giới hạn số review mỗi split.
    """
    fix_ssl()
    from datasets import load_dataset
    # Đề yêu cầu load_dataset("imdb"). Với huggingface_hub mới, id ngắn "imdb"
    # không còn phân giải được; "stanfordnlp/imdb" là CÙNG dataset (25k train / 25k test).
    try:
        ds = load_dataset("imdb")
    except Exception:
        ds = load_dataset("stanfordnlp/imdb")
    texts, labels = ds["train"]["text"], np.array(ds["train"]["label"])
    test_texts, test_labels = ds["test"]["text"], np.array(ds["test"]["label"])

    perm = np.random.RandomState(C.SPLIT_SEED).permutation(len(texts))
    n_val = int(len(texts) * C.VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(C.RESULTS_DIR / "split_indices.json", "w") as f:
        json.dump({"seed": C.SPLIT_SEED, "train": train_idx.tolist(),
                   "val": val_idx.tolist()}, f)

    if limit:  # chế độ chạy thử
        train_idx, val_idx = train_idx[:limit], val_idx[:limit // 4]
    splits = {
        "train": ([texts[i] for i in train_idx], labels[train_idx]),
        "val": ([texts[i] for i in val_idx], labels[val_idx]),
        "test": (test_texts, test_labels),
    }
    if limit:
        rs = np.random.RandomState(0).permutation(len(test_texts))[:limit // 2]
        splits["test"] = ([test_texts[i] for i in rs], test_labels[rs])

    data = {}
    for name, (t, y) in splits.items():
        data[name] = {"texts": t, "labels": np.asarray(y, dtype=np.int64),
                      "tokens": [tokenize(x) for x in t]}
    return data


def build_vocab(train_tokens, min_freq=C.MIN_FREQ):
    """Vocabulary chỉ từ TRAIN. id 0 = <pad>, id 1 = <unk>, rồi các từ theo tần suất giảm dần."""
    counts = Counter(tok for toks in train_tokens for tok in toks)
    words = [w for w, c in counts.most_common() if c >= min_freq]
    vocab = {C.PAD: C.PAD_ID, C.UNK: C.UNK_ID}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode_split(token_lists, vocab, max_len=C.MAX_LEN):
    """Đổi token -> id, cắt về max_len (giữ phần ĐẦU), pad bên PHẢI bằng <pad>.

    Trả về (ids [N, max_len], lengths [N]); lengths là độ dài thật (>=1) để mô
    hình lấy đúng trạng thái ẩn cuối và padding không trở thành tín hiệu học.
    """
    ids = np.zeros((len(token_lists), max_len), dtype=np.int64)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        row = [vocab.get(t, C.UNK_ID) for t in toks[:max_len]]
        if not row:               # review rỗng sau tokenize: coi như 1 token <unk>
            row = [C.UNK_ID]
        ids[i, :len(row)] = row
        lengths[i] = len(row)
    return ids, lengths


def unk_stats(token_lists, vocab):
    """Tỉ lệ token (toàn bộ review, chưa cắt) và tỉ lệ từ vựng duy nhất không có trong vocab."""
    total = unk = 0
    uniq, uniq_unk = set(), set()
    for toks in token_lists:
        for t in toks:
            total += 1
            uniq.add(t)
            if t not in vocab:
                unk += 1
                uniq_unk.add(t)
    return {"tokens": total, "unk_token_pct": 100 * unk / total,
            "unique_types": len(uniq), "unk_type_pct": 100 * len(uniq_unk) / len(uniq)}


def review_unk_rates(token_lists, vocab):
    """Tỉ lệ <unk> của TỪNG review (dùng để chọn tập con review nhiều từ hiếm/chưa thấy)."""
    return np.array([sum(t not in vocab for t in toks) / max(len(toks), 1)
                     for toks in token_lists])
