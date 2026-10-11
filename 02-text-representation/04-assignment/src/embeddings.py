"""Word2Vec CBOW, GloVe, fastText CBOW và căn chỉnh (align) vào cùng một vocabulary.

Ba phương pháp học vector theo cách khác nhau:
  - Word2Vec CBOW : đoán TỪ ĐÍCH từ các từ ngữ cảnh xung quanh (cửa sổ cục bộ).
  - GloVe         : khớp vector với thống kê ĐỒNG XUẤT HIỆN TOÀN CỤC của cặp từ
                    (KHÔNG phải CBOW). Ở đây dùng vector công khai đã huấn luyện
                    sẵn trên Wikipedia + Gigaword (corpus KHÁC IMDB).
  - fastText CBOW : như CBOW nhưng mỗi từ = từ + các n-gram ký tự (3..6), nên
                    có thể tạo vector cho từ chưa từng thấy từ các n-gram con.

Rò rỉ nhãn: Word2Vec và fastText chỉ được huấn luyện trên TOKEN của review
TRAIN (không nhãn, không validation, không test).
"""
import json
import time

import numpy as np

from . import config as C
from .data import fix_ssl


# --- Huấn luyện / nạp -------------------------------------------------------
def train_word2vec(train_tokens):
    """Word2Vec CBOW (sg=0), negative sampling, dim 100, window 5, min_count 2."""
    from gensim.models import Word2Vec
    path = C.CACHE_DIR / "word2vec_cbow.model"
    meta = C.CACHE_DIR / "word2vec_cbow.json"
    if path.exists():
        return Word2Vec.load(str(path)), json.load(open(meta))
    t0 = time.time()
    model = Word2Vec(
        sentences=train_tokens, vector_size=C.EMB_DIM, window=C.W2V_WINDOW,
        min_count=C.W2V_MIN_COUNT, sg=0,              # sg=0 -> CBOW
        hs=0, negative=C.W2V_NEGATIVE,                # negative sampling
        epochs=C.W2V_EPOCHS, workers=4, seed=C.EMB_SEED)
    info = {"method": "word2vec_cbow", "train_time_s": time.time() - t0,
            "vocab": len(model.wv)}
    C.CACHE_DIR.mkdir(exist_ok=True, parents=True)
    model.save(str(path))
    json.dump(info, open(meta, "w"))
    return model, info


def train_fasttext(train_tokens):
    """fastText CBOW (sg=0) với n-gram ký tự min_n=3, max_n=6."""
    from gensim.models import FastText
    path = C.CACHE_DIR / "fasttext_cbow.model"
    meta = C.CACHE_DIR / "fasttext_cbow.json"
    if path.exists():
        return FastText.load(str(path)), json.load(open(meta))
    t0 = time.time()
    model = FastText(
        sentences=train_tokens, vector_size=C.EMB_DIM, window=C.W2V_WINDOW,
        min_count=C.W2V_MIN_COUNT, sg=0, hs=0, negative=C.W2V_NEGATIVE,
        epochs=C.W2V_EPOCHS, min_n=C.FT_MIN_N, max_n=C.FT_MAX_N,
        workers=4, seed=C.EMB_SEED)
    info = {"method": "fasttext_cbow", "train_time_s": time.time() - t0,
            "vocab": len(model.wv)}
    C.CACHE_DIR.mkdir(exist_ok=True, parents=True)
    model.save(str(path))
    json.dump(info, open(meta, "w"))
    return model, info


def load_glove():
    """GloVe 100 chiều từ gensim-data (glove-wiki-gigaword-100). Tải lần đầu ~128 MB."""
    fix_ssl()
    import gensim.downloader as api
    t0 = time.time()
    kv = api.load(C.GLOVE_NAME)
    return kv, {"method": "glove", "train_time_s": 0.0,   # không huấn luyện, chỉ nạp
                "load_time_s": time.time() - t0, "vocab": len(kv)}


# --- Căn chỉnh vào vocabulary chung ------------------------------------------
def build_embedding_matrix(vocab, vector_lookup, dim=C.EMB_DIM, seed=0):
    """Tạo ma trận [vocab_size, dim] cho một nguồn embedding.

    - Bắt đầu bằng ma trận ngẫu nhiên nhỏ N(0, INIT_STD).
    - Hàng của từ có trong `vector_lookup` được ghi đè bằng vector thật.
    - Hàng <pad> = 0.
    `vector_lookup(token)` trả về vector hoặc None nếu nguồn không có từ đó.
    Trả về (matrix float32, found [vocab_size] bool, thống kê coverage).
    """
    rng = np.random.RandomState(seed)
    matrix = rng.normal(0.0, C.INIT_STD, size=(len(vocab), dim)).astype(np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    for tok, idx in vocab.items():
        if idx in (C.PAD_ID, C.UNK_ID):
            continue
        vec = vector_lookup(tok)
        if vec is not None:
            matrix[idx] = vec
            found[idx] = True
    matrix[C.PAD_ID] = 0.0
    n_words = len(vocab) - 2
    stats = {"vocab_words": n_words, "found": int(found.sum()),
             "unique_coverage_pct": 100 * found.sum() / n_words}
    return matrix, found, stats


def random_matrix(vocab, dim=C.EMB_DIM, seed=0):
    """E0: toàn bộ vector là ngẫu nhiên (control), <pad> = 0."""
    m = np.random.RandomState(seed).normal(0.0, C.INIT_STD, (len(vocab), dim)).astype(np.float32)
    m[C.PAD_ID] = 0.0
    return m


def kv_lookup(kv):
    """Bọc KeyedVectors thành hàm tra từ (None nếu từ không có trong từ vựng tường minh)."""
    index = kv.key_to_index
    return lambda tok: kv[tok] if tok in index else None


def token_coverage(token_lists, vocab, found):
    """% token (toàn bộ review) mà từ đó có trong vocab VÀ nguồn có vector thật."""
    total = hit = 0
    for toks in token_lists:
        for t in toks:
            total += 1
            idx = vocab.get(t)
            if idx is not None and found[idx]:
                hit += 1
    return 100 * hit / total


# --- Minh họa khả năng của fastText với từ chưa thấy -------------------------
def subword_demo(w2v, ft, words=("terrrible", "amazinggg", "unwatchable", "awsome", "movvie")):
    """So sánh: Word2Vec không có vector cho từ lạ, fastText vẫn tạo được từ n-gram."""
    lines = ["Từ lạ / sai chính tả: Word2Vec so với fastText\n"]
    for w in words:
        in_w2v = w in w2v.wv.key_to_index
        in_ft = w in ft.wv.key_to_index
        vec = ft.wv.get_vector(w)           # luôn tạo được từ n-gram ký tự
        neigh = [n for n, _ in ft.wv.most_similar(w, topn=5)]
        lines.append(f"- '{w}': word2vec có vector? {in_w2v} | fastText có từ tường minh? "
                     f"{in_ft} | fastText tạo vector từ n-gram: ‖v‖={np.linalg.norm(vec):.2f}"
                     f" | gần nhất: {neigh}")
    return "\n".join(lines)
