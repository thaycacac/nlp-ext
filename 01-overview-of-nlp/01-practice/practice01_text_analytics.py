"""
PRACTICE 1 - TEXT ANALYTICS (Tổng quan về NLP)
==============================================

Các tác vụ (theo slide):
    1. Tokenization          (tách từ)
    2. Stopword removal      (loại bỏ từ dừng)
    3. Lemmatization         (đưa từ về dạng gốc)
    4. TF-IDF Vectorization  (biến văn bản thành vector số)
dùng thư viện Gensim (+ NLTK cho bước lemmatization, xem ghi chú ở bước 3).

Cách chạy (từ thư mục gốc repo, dùng môi trường ảo .venv ở gốc):
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # lần đầu
    .venv/bin/python 01-overview-of-nlp/01-practice/practice01_text_analytics.py

--------------------------------------------------------------------------
LỊCH SỬ PHÁT TRIỂN CỦA NLP (để biết các kỹ thuật dưới đây xuất phát từ đâu)
--------------------------------------------------------------------------
Thập niên 1950  -  Thời kỳ luật (rule-based / symbolic)
       - 1950: Turing đề xuất "Turing test".
       - 1954: thí nghiệm Georgetown-IBM, demo dịch máy đầu tiên
         (Nga -> Anh) bằng các luật viết tay.
       - 1966: ELIZA, chatbot dùng luật so khớp mẫu câu.
       Ý tưởng: nhà ngôn ngữ học tự viết luật ngữ pháp. Dễ vỡ, khó mở rộng.

Thập niên 1980-2000  -  Thời kỳ thống kê (statistical)
       - Luật được thay bằng xác suất học từ kho ngữ liệu (corpus):
         mô hình ngôn ngữ n-gram, Hidden Markov Model (gán nhãn từ loại),
         Naive Bayes / SVM để phân loại, mô hình dịch thống kê của IBM.
       - ĐÂY là nơi "traditional NLP pipeline" trong slide ra đời:
         tiền xử lý -> trích xuất đặc trưng (Bag-of-Words, TF-IDF)
         -> chọn đặc trưng -> khai phá văn bản -> đánh giá.
         Các mốc: TF-IDF (Sparck Jones, 1972), thuật toán stemming Porter
         (1980), LSA (1990), mô hình chủ đề LDA (2003).

2013-2017  -  Thời kỳ mạng nơ-ron / word embedding
       - word2vec (2013), GloVe (2014): mỗi từ thành một vector dày (dense)
         nắm được ngữ nghĩa ("king - man + woman ~ queen"). Cách này khắc
         phục điểm yếu chính của TF-IDF (coi "car" và "automobile" là hai
         từ hoàn toàn khác nhau).
       - RNN / LSTM / seq2seq có attention dùng cho dịch máy.

2017 đến nay  -  Thời kỳ Transformer / LLM
       - 2017: bài báo "Attention Is All You Need" (kiến trúc Transformer).
       - 2018: BERT, GPT -> pre-train trên lượng văn bản khổng lồ rồi
         fine-tune cho từng bài toán.
       - 2020+: GPT-3/4, ChatGPT, Claude, Gemini ... (các ví dụ "trò chuyện
         như người", "image captioning", "image generator" trong slide).
         Tiền xử lý thủ công phần lớn được thay bằng subword tokenizer
         (BPE / WordPiece) và embedding học được.

Vì sao vẫn học pipeline cổ điển? TF-IDF + bộ phân loại đơn giản vừa nhanh,
rẻ, dễ giải thích, vừa là baseline mạnh. Và mọi bước dưới đây (tách từ,
chuẩn hóa, vector hóa) vẫn tồn tại ở dạng nào đó trong các hệ thống hiện đại.
"""

import math
import ssl

import certifi
import nltk
from gensim import corpora
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, remove_stopwords
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Python bản python.org trên macOS thường thiếu chứng chỉ SSL gốc, khiến
# nltk.download báo lỗi CERTIFICATE_VERIFY_FAILED. Ta trỏ SSL sang certifi.
ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
    cafile=certifi.where())

# NLTK chỉ cài phần code; dữ liệu ngôn ngữ (từ điển WordNet, bộ gán nhãn
# từ loại) phải tải về một lần. quiet=True để output gọn.
for resource in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"):
    nltk.download(resource, quiet=True)


def banner(title: str) -> None:
    """In tiêu đề ngăn cách giữa các bước cho dễ đọc output."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 0. BỘ DỮ LIỆU (CORPUS)
# ---------------------------------------------------------------------------
# "Corpus" = tập hợp các "document" (văn bản). Ở đây mỗi review là 1 document.
# Đây là 3 review pizza trong ví dụ TF-IDF của slide. Dữ liệu rất nhỏ để
# bạn có thể tự tính tay và đối chiếu từng con số.
reviews = [
    "The pizza was delicious and cheesy",                      # R1
    "The service was slow and the pizza was cold",             # R2
    "Delicious service and amazing experience",                # R3
]

banner("0. RAW DOCUMENTS")
for i, text in enumerate(reviews, 1):
    print(f"R{i}: {text}")


# ---------------------------------------------------------------------------
# 1. TOKENIZATION  (slide: "Word segmentation")
# ---------------------------------------------------------------------------
# Máy tính chỉ thấy một chuỗi ký tự, không thấy "từ". Tokenization cắt chuỗi
# thành các token (ở đây là từ) để có thể đếm và so sánh.
#
# gensim.utils.simple_preprocess làm 3 việc cùng lúc:
#   - tách theo ký tự không phải chữ cái (nên dấu câu biến mất)
#   - chuyển về chữ thường ("The" và "the" thành cùng một token)
#   - bỏ token quá ngắn / quá dài (mặc định < 2 hoặc > 15 ký tự)
#
# Lưu ý: tokenization thực tế khó hơn nhiều (tiếng Anh "don't", "New York";
# tiếng Trung / tiếng Việt không có khoảng trắng phân tách từ ghép...).
# Các LLM hiện đại dùng subword tokenizer (BPE) thay vì tách theo từ.
banner("1. TOKENIZATION (split + lowercase)")
tokenized = [simple_preprocess(doc) for doc in reviews]
for i, tokens in enumerate(tokenized, 1):
    print(f"R{i}: {tokens}")


# ---------------------------------------------------------------------------
# 2. STOPWORD REMOVAL (loại bỏ từ dừng)
# ---------------------------------------------------------------------------
# Stopword là các từ chức năng rất phổ biến ("the", "was", "and"), xuất hiện
# ở hầu hết văn bản nên gần như không giúp phân biệt văn bản này với văn bản
# khác. Loại bỏ chúng giúp:
#   + thu nhỏ từ vựng (giảm số chiều - xem slide "Why text is tough")
#   + làm nổi bật các từ mang thông tin
# Cẩn thận: với một số bài toán (vd. sentiment: "not good") stopword lại quan
# trọng, bỏ đi sẽ mất nghĩa!
banner("2. STOPWORD REMOVAL")
print(f"Gensim có sẵn {len(STOPWORDS)} stopword, ví dụ:",
      sorted(STOPWORDS)[:8], "...")

# remove_stopwords() chỉ nhận chuỗi, nên ở đây ta lọc trực tiếp list token
# bằng cùng tập STOPWORDS (frozenset -> tra cứu O(1)).
no_stop = [[t for t in tokens if t not in STOPWORDS] for tokens in tokenized]
for i, tokens in enumerate(no_stop, 1):
    print(f"R{i}: {tokens}")

# (Phiên bản dùng chuỗi, tiện cho các thử nghiệm nhanh:)
print("Demo remove_stopwords():", remove_stopwords("the pizza was delicious"))


# ---------------------------------------------------------------------------
# 3. LEMMATIZATION (so sánh với stemming)
# ---------------------------------------------------------------------------
# Mục tiêu: đưa các dạng khác nhau của một từ về MỘT dạng gốc để được đếm
# chung ("running", "ran", "runs" -> "run"). Giúp giảm số chiều.
#
#   Stemming      : cắt hậu tố theo luật (Porter, 1980). Nhanh nhưng thô;
#                   kết quả có thể không phải từ thật: delicious -> "delici".
#                   (Slide dùng cách này.)
#   Lemmatization : tra từ điển (WordNet) và trả về dạng gốc thật (lemma):
#                   better -> good, was -> be. Chậm hơn nhưng đúng ngôn ngữ
#                   hơn. Cần biết từ loại (POS) mới chính xác: "meeting" có
#                   thể là danh từ (cuộc họp) hoặc động từ (gặp, meet).
#
# LƯU Ý: Gensim 4 đã bỏ hàm lemmatize riêng (trước đây dựa vào thư viện
# `pattern`), nên ta dùng WordNetLemmatizer của NLTK cho đúng bước này và
# vẫn dùng Gensim cho các bước còn lại.
banner("3. LEMMATIZATION (compared with stemming)")


def to_wordnet_pos(treebank_tag: str) -> str:
    """Đổi nhãn POS của Penn Treebank (từ nltk.pos_tag) sang nhãn của WordNet."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # mặc định: danh từ


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def lemmatize(tokens):
    # Gán nhãn từ loại (POS tagging, có trong slide) trước, vì lemma phụ
    # thuộc vào từ loại. Gán nhãn chính xác nhất khi có cả câu, nhưng với
    # demo nhỏ này thế là đủ.
    return [lemmatizer.lemmatize(tok, to_wordnet_pos(tag))
            for tok, tag in nltk.pos_tag(tokens)]


lemmatized = [lemmatize(tokens) for tokens in no_stop]
for i, (before, lem) in enumerate(zip(no_stop, lemmatized), 1):
    stem = [stemmer.stem(t) for t in before]
    print(f"R{i}: tokens  = {before}")
    print(f"     lemmas  = {lem}")
    print(f"     stems   = {stem}   <- cách của slide")

# Quan sát: "amazing" thành "amaze" - bộ gán nhãn đoán nó là ĐỘNG TỪ vì từ
# đứng riêng lẻ không có ngữ cảnh. Nếu đưa cả câu vào, nó sẽ được gán là
# tính từ và giữ nguyên "amazing". Lemmatization chỉ tốt bằng chất lượng
# của POS tag.

# Demo nhỏ cho thấy lemmatization mạnh hơn stemming (cần chỉ rõ từ loại):
print("\nDemo:", [(w, lemmatizer.lemmatize(w, "v")) for w in
                  ["running", "ran", "was", "flies"]],
      "| better (adj) ->", lemmatizer.lemmatize("better", "a"))


# ---------------------------------------------------------------------------
# 4. TF-IDF VECTORIZATION
# ---------------------------------------------------------------------------
# Máy tính cần con số. Bước A: Bag of Words (BoW) - đếm số lần mỗi từ xuất
# hiện trong mỗi văn bản, bỏ qua thứ tự từ (slide: "Feature Extraction -
# Indexing"). Bước B: TF-IDF đánh trọng số lại các số đếm đó:
#
#   TF  (term frequency)          = số lần từ xuất hiện trong văn bản NÀY
#   IDF (inverse doc. frequency)  = log(N / df)
#                                   N  = tổng số văn bản
#                                   df = số văn bản chứa từ đó
#   TF-IDF = TF * IDF
#
# Trực giác: từ xuất hiện ở mọi văn bản (df = N) có IDF = log(1) = 0 ->
# trọng số 0 (vô dụng để phân biệt). Từ hiếm nhưng xuất hiện nhiều trong
# một văn bản sẽ có trọng số cao -> "từ khóa đặc trưng" (signal word).
banner("4. TF-IDF VECTORIZATION")

# Dictionary: ánh xạ mỗi token duy nhất -> một số nguyên id (chính là chiều
# của vector).
dictionary = corpora.Dictionary(lemmatized)
print("Dictionary (token -> id):", dict(dictionary.token2id))
print("Document frequency (id -> df):", dict(dictionary.dfs))

# Bag of words: mỗi văn bản thành danh sách THƯA các cặp (token_id, số lần).
# Token có số lần = 0 không được lưu (văn bản thật phần lớn là 0).
bow_corpus = [dictionary.doc2bow(doc) for doc in lemmatized]
print("\nBag of words (id, count):")
for i, bow in enumerate(bow_corpus, 1):
    print(f"R{i}: {bow}  ->  "
          f"{[(dictionary[i_], c) for i_, c in bow]}")

# --- 4a. TF-IDF đúng như công thức slide: tf * ln(N / df) ------------------
# Mặc định của Gensim khác slide: dùng log CƠ SỐ 2 và chuẩn hóa L2 từng
# vector. Để khớp công thức slide, ta truyền hàm trọng số toàn cục riêng
# (log tự nhiên) và tắt chuẩn hóa.
tfidf_slide = TfidfModel(
    bow_corpus,
    wglobal=lambda df, total_docs: math.log(total_docs / df),
    normalize=False,
)

print("\n--- TF-IDF (công thức slide: tf * ln(N/df), không chuẩn hóa) ---")
for i, vec in enumerate(tfidf_slide[bow_corpus], 1):
    print(f"R{i}: " + ", ".join(f"{dictionary[t]}={w:.3f}" for t, w in vec))

# --- 4b. TF-IDF mặc định của Gensim (log2 + chuẩn hóa L2) ------------------
# Chuẩn hóa L2 làm mọi vector có độ dài 1, nên văn bản dài và ngắn so sánh
# được với nhau (cosine similarity khi đó chỉ là tích vô hướng).
tfidf = TfidfModel(bow_corpus)
print("\n--- TF-IDF (mặc định Gensim: log2, chuẩn hóa L2) ---")
for i, vec in enumerate(tfidf[bow_corpus], 1):
    print(f"R{i}: " + ", ".join(f"{dictionary[t]}={w:.3f}" for t, w in vec))

# --- 4c. Bảng dày: văn bản là vector (slide "Reviews as vectors") ----------
# Mỗi văn bản là một điểm trong không gian mà các trục là các từ vựng.
# Từ vựng thật có 20.000+ trục, vì vậy người ta lưu vector dạng thưa.
vocab = [dictionary[i] for i in range(len(dictionary))]
print("\n--- Ma trận document-term (TF-IDF theo công thức slide) ---")
print(f"{'':4}" + "".join(f"{w:>12}" for w in vocab))
for i, vec in enumerate(tfidf_slide[bow_corpus], 1):
    weights = dict(vec)
    print(f"R{i:<3}" + "".join(f"{weights.get(j, 0.0):>12.3f}"
                              for j in range(len(vocab))))

# --- 4d. Tự kiểm tra câu hỏi của slide: TF-IDF của "delici(ous)" -----------
# "delicious" xuất hiện ở R1 và R3 (df = 2), N = 3  ->  IDF = ln(3/2) = 0.405
term = "delicious"
df = dictionary.dfs[dictionary.token2id[term]]
idf = math.log(len(reviews) / df)
print(f"\nTính tay cho '{term}': N={len(reviews)}, df={df}, "
      f"IDF=ln({len(reviews)}/{df})={idf:.3f}")
for i, lem in enumerate(lemmatized, 1):
    tf = lem.count(term)
    print(f"  R{i}: TF={tf}  TF-IDF={tf * idf:.3f}")

# Rút ra: "pizza", "delicious", "service" (df=2) phổ biến ngang nhau, còn
# "cheesy", "slow", "cold", "amaze", "experience" (df=1) có IDF cao nhất
# = ln(3) = 1.099. TF-IDF vẫn là bag of words: không có thứ tự từ và không
# hiểu nghĩa ("good" và "great" bị coi là không liên quan) - đúng là điều
# word2vec và Transformer sau này đã khắc phục.
