"""
EXERCISE 6 (ADVANCED) - SENTENCE EMBEDDINGS
===========================================
Biến cả CÂU thành MỘT vector bằng cách lấy trung bình vector của các token
(mean pooling), rồi so sánh độ giống nhau về ngữ nghĩa bằng cosine:
    1. "He went to the bank to get cash."
    2. "She deposited money in the bank."
    3. "They sat on the river bank."

Cách chạy (từ thư mục gốc repo):
    .venv/bin/python 02-text-representation/03-sentence-embeddings/exercise06_sentence_embeddings.py

--------------------------------------------------------------------------
BỐI CẢNH: TỪ EMBEDDING TỪ ĐẾN EMBEDDING CÂU / TÀI LIỆU
--------------------------------------------------------------------------
Nhiều bài toán (tìm kiếm, phân cụm, hỏi đáp, truy xuất) cần so sánh CẢ CÂU
hay cả tài liệu, không chỉ từng từ. Cách đơn giản nhất là trung bình embedding
các từ (cũng làm được với Word2Vec/GloVe, nhưng mất ngữ cảnh). Với BERT, mỗi
token đã mang ngữ cảnh nên trung bình có ý nghĩa hơn.
Cách tốt hơn: Sentence-BERT (2019) fine-tune BERT bằng cặp câu giống/khác
nhau để vector câu thật sự phản ánh ngữ nghĩa; sau đó là Universal Sentence
Encoder và các mô hình embedding hiện đại dùng cho RAG / semantic search.
"""

import itertools

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-uncased"

# --- Bước 1: ba câu -----------------------------------------------------------
SENTENCES = {
    "S1": "He went to the bank to get cash.",       # nghĩa tài chính
    "S2": "She deposited money in the bank.",       # nghĩa tài chính
    "S3": "They sat on the river bank.",            # nghĩa bờ sông
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


@torch.no_grad()
def sentence_embedding(sentence, include_special_tokens=True):
    """Embedding câu = trung bình vector của các token (mean pooling).

    - Mỗi token có một vector 768 chiều (last_hidden_state).
    - Trung bình theo chiều "số token" -> một vector 768 chiều cho cả câu.
    - include_special_tokens=True: tính cả [CLS] và [SEP] (đúng như đề: "all
      tokens"). Đặt False để chỉ tính các token thật của câu.
    Ở đây mỗi lần chỉ xử lý MỘT câu nên không có padding. Khi xử lý theo lô
    (batch) phải nhân với attention_mask để padding không bị tính vào trung bình.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    hidden = model(**inputs).last_hidden_state[0]          # (số_token, 768)
    if not include_special_tokens:
        hidden = hidden[1:-1]                               # bỏ [CLS] và [SEP]
    return hidden.mean(dim=0)                               # (768,)


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


# --- Bước 2: embedding cho từng câu -------------------------------------------
print("=" * 70)
print("1. Sentence embeddings (mean pooling của BERT)")
print("=" * 70)
for name, text in SENTENCES.items():
    tokens = tokenizer.tokenize(text)
    print(f"{name}: \"{text}\"  ->  {len(tokens)} token + [CLS] + [SEP]")


def compare(include_special_tokens):
    """Tính cosine cho ba cặp câu, trả về dict {cặp: giá trị}."""
    embs = {n: sentence_embedding(t, include_special_tokens)
            for n, t in SENTENCES.items()}
    return {f"{a} & {b}": cosine(embs[a], embs[b])
            for a, b in itertools.combinations(SENTENCES, 2)}


# --- Bước 3: cosine similarity giữa các cặp -----------------------------------
labels = {
    "S1 & S2": "tài chính - tài chính",
    "S1 & S3": "tài chính - sông",
    "S2 & S3": "tài chính - sông",
}
results = {True: compare(True), False: compare(False)}

print("\n" + "=" * 70)
print("2. Cosine similarity giữa các câu")
print("=" * 70)
print(f"{'cặp':<10}{'ý nghĩa':<24}{'tính cả [CLS]/[SEP]':>22}{'bỏ [CLS]/[SEP]':>18}")
for pair, meaning in labels.items():
    print(f"{pair:<10}{meaning:<24}{results[True][pair]:>22.4f}"
          f"{results[False][pair]:>18.4f}")

# --- Bước 4: so sánh -----------------------------------------------------------
print("\n" + "=" * 70)
print("3. KẾT LUẬN")
print("=" * 70)
best = max(results[True], key=results[True].get)
print(f"Cặp giống nhau nhất (theo đề, tính tất cả token): {best} "
      f"= {results[True][best]:.4f}")
fin = results[True]["S1 & S2"]
mixed = (results[True]["S1 & S3"] + results[True]["S2 & S3"]) / 2
print(f"  tài chính-tài chính (S1&S2)     : {fin:.4f}")
print(f"  tài chính-sông (trung bình 2 cặp): {mixed:.4f}")
if fin > mixed:
    print("-> Hai câu cùng nghĩa 'ngân hàng' giống nhau hơn các cặp có câu về "
          "bờ sông: embedding câu bắt được sự khác biệt ngữ nghĩa.")
else:
    print("-> Kết quả KHÔNG tách rõ nghĩa tài chính và bờ sông.")

# Lưu ý quan trọng khi đọc số:
#  1. Chênh lệch thường nhỏ và mọi cặp đều có cosine cao (vd 0.7-0.9). BERT
#     gốc không được huấn luyện để vector câu so sánh bằng cosine; trung bình
#     token còn bị ảnh hưởng bởi [CLS]/[SEP] và các từ chung ("the", "to")
#     nên ba câu giống nhau một phần chỉ vì cấu trúc.
#  2. Chỉ có 3 câu nên đừng suy rộng. Hãy thử thêm câu khác để kiểm tra.
#  3. Muốn vector câu chất lượng thì dùng Sentence-BERT (thư viện
#     sentence-transformers, vd model "all-MiniLM-L6-v2"): cùng bài toán sẽ
#     cho khoảng cách giữa cặp giống/khác nghĩa rõ rệt hơn nhiều.
