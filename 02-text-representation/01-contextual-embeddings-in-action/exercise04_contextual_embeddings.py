"""
EXERCISE 4 - CONTEXTUAL EMBEDDINGS IN ACTION
============================================
Bài toán: từ "bank" xuất hiện trong 2 câu với 2 nghĩa khác nhau:
    1. "He sat on the bank of the river."          (bờ sông)
    2. "She went to the bank to deposit money."    (ngân hàng)
BERT có cho "bank" hai vector khác nhau không? Ta đo bằng cosine similarity.

Cách chạy (từ thư mục gốc repo):
    .venv/bin/python 02-text-representation/01-contextual-embeddings-in-action/exercise04_contextual_embeddings.py

--------------------------------------------------------------------------
BỐI CẢNH: TỪ EMBEDDING TĨNH ĐẾN EMBEDDING NGỮ CẢNH
--------------------------------------------------------------------------
- Word2Vec (2013), GloVe (2014), FastText (2016) là embedding TĨNH: mỗi từ
  có đúng MỘT vector cho dù xuất hiện ở đâu. "bank" (bờ sông) và "bank"
  (ngân hàng) bị nhồi chung vào một vector -> không phân biệt được đa nghĩa
  (polysemy).
- ELMo (2018, BiLSTM) rồi BERT (2018, Transformer encoder) cho embedding
  NGỮ CẢNH: vector của một từ là trạng thái ẩn của mạng khi đọc CẢ câu, nên
  mỗi lần xuất hiện của từ có một vector riêng.
- BERT được pre-train bằng Masked Language Modeling (che từ rồi đoán) và
  nhìn ngữ cảnh cả hai phía (bidirectional) nhờ cơ chế self-attention.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-uncased"   # "uncased": mọi chữ được đổi về chữ thường

SENTENCES = [
    "He sat on the bank of the river.",
    "She went to the bank to deposit money.",
]
TARGET = "bank"

# --- Bước 1: nạp BERT đã pre-train từ Hugging Face --------------------------
# Tokenizer biến chuỗi thành id; model biến id thành vector.
# model.eval() tắt dropout để kết quả ổn định giữa các lần chạy.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def find_word_positions(tokens, word):
    """Tìm vị trí của `word` trong danh sách token của BERT.

    BERT dùng WordPiece: từ lạ/hiếm bị cắt thành subword, các mảnh sau có
    tiền tố '##' (vd "playing" -> "play", "##ing"). Vì vậy một từ có thể
    chiếm NHIỀU vị trí. Ta tokenize riêng `word` để biết nó gồm mấy mảnh rồi
    tìm đoạn khớp trong câu. ("bank" là một từ phổ biến nên chỉ có 1 mảnh.)
    """
    pieces = tokenizer.tokenize(word)
    n = len(pieces)
    for start in range(len(tokens) - n + 1):
        if tokens[start:start + n] == pieces:
            return list(range(start, start + n))
    raise ValueError(f"Không tìm thấy '{word}' trong {tokens}")


@torch.no_grad()   # chỉ suy luận, không cần tính gradient -> nhanh, ít RAM
def get_word_embedding(sentence, word):
    """Trả về vector ngữ cảnh (768 chiều) của `word` trong `sentence`."""
    # Bước 2: tokenize. BERT tự thêm [CLS] ở đầu và [SEP] ở cuối câu.
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"  tokens: {tokens}")

    # Bước 3: chạy model. last_hidden_state có shape (batch, số_token, 768):
    # mỗi token một vector, đã "nhìn" toàn bộ câu qua self-attention.
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]

    # Bước 4: lấy vector tại vị trí của từ đích. Nếu từ bị cắt thành nhiều
    # subword thì lấy trung bình các vector của chúng.
    positions = find_word_positions(tokens, word)
    print(f"  '{word}' ở vị trí token {positions}")
    return hidden[positions].mean(dim=0)


def cosine(a, b):
    """Cosine similarity: 1 = cùng hướng (rất giống), 0 = vuông góc, -1 = ngược."""
    return F.cosine_similarity(a, b, dim=0).item()


# --- Chạy bài chính ----------------------------------------------------------
print("=" * 70)
print("1. BERT: vector 'bank' trong hai ngữ cảnh khác nhau")
print("=" * 70)
vectors = []
for s in SENTENCES:
    print(f"\n\"{s}\"")
    vectors.append(get_word_embedding(s, TARGET))

similarity = cosine(vectors[0], vectors[1])
print(f"\nCosine similarity (bank-sông vs bank-ngân hàng) = {similarity:.4f}")

# --- Đối chứng 1: embedding TĨNH (kiểu Word2Vec/GloVe) ----------------------
# Bảng embedding đầu vào của BERT (word_embeddings) cho mỗi token đúng một
# vector, không phụ thuộc câu. Nó giống Word2Vec ở chỗ là embedding tĩnh nên
# hai "bank" luôn giống hệt nhau (cosine = 1.0), bất kể nghĩa.
print("\n" + "=" * 70)
print("2. Đối chứng: embedding tĩnh (bảng input embedding của BERT)")
print("=" * 70)
bank_id = tokenizer.convert_tokens_to_ids(TARGET)
static_vec = model.embeddings.word_embeddings.weight[bank_id].detach()
print(f"Cosine(static bank, static bank) = {cosine(static_vec, static_vec):.4f}"
      "   <- luôn 1.0: không phân biệt được nghĩa")

# --- Đối chứng 2: hai câu CÙNG nghĩa "ngân hàng" ----------------------------
# Để kết luận "BERT phân biệt nghĩa" thật sự, ta cần thấy: cùng nghĩa -> giống
# nhau hơn, khác nghĩa -> giống nhau ít hơn.
print("\n" + "=" * 70)
print("3. Đối chứng: hai câu cùng nghĩa ngân hàng")
print("=" * 70)
same_sense = "He deposited his salary at the bank yesterday."
print(f"\n\"{same_sense}\"")
v_same = get_word_embedding(same_sense, TARGET)
sim_same = cosine(vectors[1], v_same)
print(f"\nCosine(bank-ngân hàng, bank-ngân hàng khác) = {sim_same:.4f}")

# --- Kết luận -----------------------------------------------------------------
print("\n" + "=" * 70)
print("KẾT LUẬN")
print("=" * 70)
print(f"  khác nghĩa (sông vs ngân hàng): {similarity:.4f}")
print(f"  cùng nghĩa (ngân hàng vs ngân hàng): {sim_same:.4f}")
print(f"  embedding tĩnh                : 1.0000")
if sim_same > similarity:
    print("-> Vector 'bank' của BERT THAY ĐỔI theo ngữ cảnh: hai nghĩa khác "
          "nhau cho vector khác nhau, còn cùng nghĩa thì gần nhau hơn.")
else:
    print("-> Kết quả không như kỳ vọng; hãy kiểm tra lại câu ví dụ.")
# Lưu ý: cosine giữa hai vector BERT thường không thấp hẳn (vd 0.4-0.7) dù
# khác nghĩa, vì các vector BERT chiếm một vùng hẹp trong không gian
# ("anisotropy"). Quan trọng là SO SÁNH tương đối giữa các cặp, không phải
# con số tuyệt đối.
