"""Cấu hình chung cho toàn bộ thí nghiệm.

Mọi siêu tham số nằm ở một chỗ để đảm bảo "fairness rule" của đề: tất cả điều
kiện embedding dùng CÙNG vocabulary, độ dài chuỗi, batch size, kiến trúc LSTM,
optimizer, learning rate và quy tắc early stopping.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"      # json + xác suất dự đoán của từng lần chạy
CACHE_DIR = ROOT / "cache"           # embedding đã huấn luyện, checkpoint (không commit)

# --- Dữ liệu -----------------------------------------------------------------
SPLIT_SEED = 42        # seed cố định cho chia train/validation (lưu vào split_indices.json)
VAL_FRACTION = 0.2     # 80/20 trên tập train chính thức
MAX_LEN = 300          # pad / cắt mỗi review về 300 token
MIN_FREQ = 2           # từ xuất hiện < 2 lần trong train -> <unk> (khớp min_count của w2v/fastText)
PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# --- Embedding ---------------------------------------------------------------
EMB_DIM = 100
EMB_SEED = 42          # seed huấn luyện Word2Vec / fastText (huấn luyện 1 lần, dùng cho mọi seed LSTM)
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_NEGATIVE = 5       # negative sampling
FT_MIN_N, FT_MAX_N = 3, 6
GLOVE_NAME = "glove-wiki-gigaword-100"   # gensim-data (Wikipedia 2014 + Gigaword 5, 6B token)
INIT_STD = 0.1         # độ lệch chuẩn khi khởi tạo ngẫu nhiên các hàng thiếu

# --- LSTM / huấn luyện -------------------------------------------------------
HIDDEN_DIM = 128
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 2           # early stopping theo validation loss
MAX_EPOCHS = 20        # trần số epoch (early stopping thường dừng sớm hơn)
GRAD_CLIP = 5.0        # giống nhau cho mọi run
SEEDS = (0, 1, 2)

# --- Các thí nghiệm: (id, nguồn embedding, trainable) -------------------------
EXPERIMENTS = [
    ("E0", "random",   False),
    ("E1", "word2vec", False),
    ("E2", "glove",    False),
    ("E3", "fasttext", False),
    ("E4", "word2vec", True),
    ("E4", "glove",    True),
    ("E4", "fasttext", True),
]
