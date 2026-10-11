"""
EXERCISE 5 - VISUALIZE EMBEDDINGS WITH PCA
==========================================
Lấy embedding BERT của các từ: bank, river, water, money, loan, cash
(đặt trong câu để BERT có ngữ cảnh), giảm xuống 2 chiều bằng PCA rồi vẽ để
xem có tách thành cụm "tài chính" và "thiên nhiên" hay không.

Cách chạy (từ thư mục gốc repo):
    .venv/bin/python 02-text-representation/02-visualize-embeddings-with-pcat-sne/exercise05_visualize_embeddings.py
    (thêm --show để mở cửa sổ đồ thị; mặc định chỉ lưu ra file PNG)

--------------------------------------------------------------------------
VÌ SAO PHẢI GIẢM CHIỀU?
--------------------------------------------------------------------------
Embedding BERT có 768 chiều - con người không "nhìn" được. PCA (Principal
Component Analysis) tìm 2 hướng mà dữ liệu biến thiên nhiều nhất rồi chiếu
lên đó, giữ lại càng nhiều cấu trúc càng tốt. t-SNE là lựa chọn khác, bảo
toàn quan hệ lân cận cục bộ nhưng cần nhiều điểm hơn (perplexity phải nhỏ hơn
số điểm) nên với 7 điểm như ở đây PCA hợp lý hơn.
"""

import os
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")   # không mở cửa sổ, chỉ lưu file (chạy được trên server)
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-uncased"

# --- Bước 1+2: bộ từ và câu chứa chúng ----------------------------------------
# Hai câu: một về tài chính, một về thiên nhiên. "bank" xuất hiện trong CẢ HAI
# câu - ta lấy cả hai vector để xem BERT đặt chúng ở đâu. Mỗi phần tử:
# (câu, từ đích, nhãn trên đồ thị, nhóm để tô màu).
SENT_FINANCE = "She went to the bank to get a loan and withdraw some cash and money."
SENT_NATURE = "He sat on the bank of the river and watched the water flow."
ITEMS = [
    (SENT_FINANCE, "bank",  "bank (finance)", "finance"),
    (SENT_FINANCE, "money", "money",          "finance"),
    (SENT_FINANCE, "loan",  "loan",           "finance"),
    (SENT_FINANCE, "cash",  "cash",           "finance"),
    (SENT_NATURE,  "bank",  "bank (river)",   "nature"),
    (SENT_NATURE,  "river", "river",          "nature"),
    (SENT_NATURE,  "water", "water",          "nature"),
]
COLORS = {"finance": "tab:red", "nature": "tab:green"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


@torch.no_grad()
def get_word_embedding(sentence, word):
    """Vector ngữ cảnh 768 chiều của `word` trong `sentence` (lớp cuối của BERT)."""
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Từ có thể bị WordPiece cắt thành nhiều mảnh ("##..."): tìm đoạn khớp.
    pieces = tokenizer.tokenize(word)
    for start in range(len(tokens) - len(pieces) + 1):
        if tokens[start:start + len(pieces)] == pieces:
            positions = list(range(start, start + len(pieces)))
            break
    else:
        raise ValueError(f"Không thấy '{word}' trong {tokens}")
    hidden = model(**inputs).last_hidden_state[0]
    return hidden[positions].mean(dim=0)


# --- Bước 3: lấy embedding cho từng từ ----------------------------------------
vectors = torch.stack([get_word_embedding(s, w) for s, w, _, _ in ITEMS])
print(f"Ma trận embedding: {tuple(vectors.shape)}  (số từ x 768 chiều)")

# --- Bước 4: PCA 768 chiều -> 2 chiều -----------------------------------------
pca = PCA(n_components=2)
points = pca.fit_transform(vectors.numpy())
ratio = pca.explained_variance_ratio_
print(f"Phương sai giữ lại: PC1={ratio[0]:.1%}, PC2={ratio[1]:.1%} "
      f"(tổng {ratio.sum():.1%})")

# --- Bước 5+6: vẽ và gắn nhãn từng điểm ---------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for (_, _, label, group), (x, y) in zip(ITEMS, points):
    ax.scatter(x, y, c=COLORS[group], s=80)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))
    print(f"  {label:<14} ({x:+.2f}, {y:+.2f})")
ax.scatter([], [], c=COLORS["finance"], label="finance")
ax.scatter([], [], c=COLORS["nature"], label="nature")
ax.legend()
ax.set_title("BERT contextual embeddings (PCA 2D)")
ax.set_xlabel(f"PC1 ({ratio[0]:.0%})")
ax.set_ylabel(f"PC2 ({ratio[1]:.0%})")
ax.grid(alpha=0.3)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "embeddings_pca.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nĐã lưu đồ thị: {out_path}")
if "--show" in sys.argv:
    plt.show()

# --- Cách đọc kết quả ---------------------------------------------------------
# - Kỳ vọng: money / loan / cash nằm gần nhau (cụm tài chính); river / water
#   gần nhau (cụm thiên nhiên); "bank (finance)" về phía cụm tài chính còn
#   "bank (river)" về phía cụm thiên nhiên. Đây chính là điểm mạnh của
#   embedding ngữ cảnh: cùng chữ "bank" nhưng hai vị trí khác nhau.
# - Cảnh báo: chỉ có 7 điểm và mỗi nhóm đến từ MỘT câu, nên một phần khoảng
#   cách là do "các từ cùng câu thì giống nhau" chứ không hoàn toàn do nghĩa.
#   Muốn kết luận chắc hơn hãy thêm nhiều câu cho mỗi từ.
