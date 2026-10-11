"""
PRACTICE 2 - PREPROCESSING SOCIAL MEDIA DATA
============================================

Các tác vụ (theo slide):
    1. Noise Removal          (xóa URL, mention, hashtag; quyết định về emoji)
    2. Tokenization           (so sánh tách theo khoảng trắng và tokenizer nâng cao)
    3. Normalization          (chữ thường, ký tự lặp, từ viết tắt / tiếng lóng)
    4. Stopword Handling      (loại stopword + thảo luận cho sentiment analysis)
    5. Stemming vs Lemmatization
    6. Final Cleaned Data     (token sạch, sẵn sàng đưa vào mô hình ML)

Cách chạy (từ thư mục gốc repo, dùng môi trường ảo .venv ở gốc):
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # lần đầu
    .venv/bin/python 01-overview-of-nlp/02-practice/practice02_social_media_preprocessing.py

--------------------------------------------------------------------------
VÌ SAO DỮ LIỆU MẠNG XÃ HỘI KHÓ HƠN VĂN BẢN THƯỜNG?
--------------------------------------------------------------------------
Pipeline ở practice 01 giả định văn bản "sạch" (review viết đúng chính tả).
Bài đăng mạng xã hội thì ngược lại: có URL, @mention, #hashtag, emoji,
kéo dài chữ ("looooveee"), viết tắt ("thx", "rn", "u"), sai chính tả cố ý...
Đây là lý do NLP thời kỳ thống kê (2000s) phải có bước chuẩn hóa rất kỹ.
Các mô hình hiện đại (BERTweet, twitter-roberta...) được pre-train trực tiếp
trên tweet nên vẫn cần vài bước tối thiểu: thay URL bằng "http" và mention
bằng "@user", rồi để subword tokenizer tự xử lý phần còn lại.
"""

import itertools
import re
import ssl
import unicodedata

import certifi
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize

# Sửa lỗi SSL của Python bản python.org trên macOS (xem practice 01).
ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
    cafile=certifi.where())

for resource in ("wordnet", "omw-1.4", "stopwords", "punkt_tab",
                 "averaged_perceptron_tagger_eng"):
    nltk.download(resource, quiet=True)


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label: str, data) -> None:
    """In kết quả của một bước cho từng bài đăng."""
    print(f"\n[{label}]")
    for i, item in enumerate(data, 1):
        print(f"  P{i}: {item}")


# ---------------------------------------------------------------------------
# 0. DATASET (5 bài đăng thô trong đề)
# ---------------------------------------------------------------------------
# Lưu ý: bài 3 dùng dấu nháy cong ’ (Can’t) như trong đề; ta sẽ chuẩn hóa
# thành nháy thẳng ' ở bước 1 để các bước sau xử lý thống nhất.
posts = [
    "OMG 😂 I looooveee this sooo muchhh!!! 🔥 🔥 🔥 #blessed #happy",
    "U r the best!! thx 4 always being there 🙏 💯",
    "Can’t wait 2 see u @JohnDoe tomorrow... http://bit.ly/123",
    "Breaking news: NEW iPhone14 Pro released!!! Check out 👉 www.apple.com",
    "Sooo tired rn... need ☕ ☕ #mondaymood",
]

banner("0. RAW POSTS")
show("raw", posts)


# ---------------------------------------------------------------------------
# 1. NOISE REMOVAL
# ---------------------------------------------------------------------------
# Xóa những thành phần không mang nghĩa ngôn ngữ cho bài toán của ta:
#   - URL      : "http://bit.ly/123", "www.apple.com" - là địa chỉ, không phải
#                nội dung; mỗi URL gần như duy nhất nên chỉ làm phình từ vựng.
#   - @mention : tên người dùng, vô nghĩa với sentiment.
#   - #hashtag : theo đề thì xóa. (Thực tế hashtag đôi khi rất giàu thông tin:
#                "#blessed" thể hiện cảm xúc tích cực. Một lựa chọn khác là
#                chỉ bỏ dấu '#' rồi giữ lại từ, hoặc tách "#mondaymood" thành
#                "monday mood". Ở đây ta làm đúng theo đề và ghi nhận sự đánh
#                đổi: bài 1 mất "blessed, happy" - hai từ rất tích cực.)
#
# Thứ tự quan trọng: xóa URL TRƯỚC, vì URL có thể chứa '@' hoặc '#' mà nếu
# xóa mention/hashtag trước sẽ cắt URL thành mảnh vụn.
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")


def remove_noise(text: str) -> str:
    text = text.replace("’", "'")          # chuẩn hóa dấu nháy cong -> thẳng
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()   # gộp khoảng trắng thừa


banner("1. NOISE REMOVAL")
no_noise = [remove_noise(p) for p in posts]
show("sau khi xóa URL / mention / hashtag", no_noise)

# --- Quyết định về EMOJI: GIỮ LẠI -------------------------------------------
# Lý do giữ:
#   + Emoji là tín hiệu cảm xúc mạnh và rất đáng tin trên mạng xã hội:
#     😂 🔥 💯 🙏 ➜ tích cực; chúng thường nói lên cảm xúc rõ hơn cả câu chữ.
#   + Bài toán đích là ML/sentiment: bỏ emoji là vứt bỏ đặc trưng tốt nhất.
#   + Với bài 2 và 5, emoji là phần lớn nội dung "cảm xúc" của bài.
# Khi nào nên XÓA: bài toán chỉ cần nội dung chủ đề (topic modelling), hoặc mô
# hình/tokenizer không hiểu emoji (sẽ thành <unk>).
# Cách trung gian: đổi emoji thành chữ (demo bên dưới) để mô hình chỉ dùng từ.
EMOJI_TO_TEXT = {
    "😂": "joy", "🔥": "fire", "🙏": "thanks", "💯": "perfect",
    "👉": "point", "☕": "coffee",
}


def is_emoji(token: str) -> bool:
    """Token là emoji/ký hiệu nếu mọi ký tự (trừ variation selector) thuộc
    nhóm Unicode 'So' (Symbol, other) - nhóm chứa hầu hết emoji."""
    chars = [c for c in token if c != "️"]
    return bool(chars) and all(unicodedata.category(c) == "So" for c in chars)


print("\nQuyết định: GIỮ emoji làm token (mang tín hiệu cảm xúc).")
print("Cách trung gian (đổi emoji -> chữ):",
      {k: v for k, v in EMOJI_TO_TEXT.items()})


# ---------------------------------------------------------------------------
# 2. TOKENIZATION - so sánh các cách tách từ
# ---------------------------------------------------------------------------
# (a) Whitespace  : str.split() - đơn giản nhất, chỉ cắt ở khoảng trắng.
#                   Dấu câu dính vào từ: "much!!!" ≠ "much"; "rn..." ≠ "rn".
# (b) NLTK word_tokenize : tokenizer cho văn bản chuẩn (Penn Treebank). Tách
#                   dấu câu và co từ ("Can't" -> "Ca", "n't"), nhưng không
#                   hiểu văn phong mạng xã hội (có thể làm vỡ "@user", "#tag").
# (c) NLTK TweetTokenizer : thiết kế riêng cho tweet; giữ nguyên emoji,
#                   emoticon (":)"), hashtag, mention, URL như MỘT token.
# (d) spaCy / HuggingFace: spaCy tokenizer theo luật (tốt cho tiếng Anh nói
#                   chung); tokenizer HuggingFace (BPE/WordPiece) cắt thành
#                   subword, vd "looooveee" -> "loo", "##oo", "##ve", "##ee":
#                   không bao giờ gặp từ lạ nhưng token khó đọc với con người.
#                   (Hai loại này nặng/phải tải thêm nên chỉ nêu ở comment.)
#
# Ta so sánh trên bài đăng THÔ (chưa xóa nhiễu) để thấy rõ khác biệt, rồi dùng
# TweetTokenizer cho pipeline vì nó xử lý emoji và dấu câu tốt nhất ở đây.
banner("2. TOKENIZATION (whitespace vs NLTK vs TweetTokenizer)")
tweet_tok = TweetTokenizer(preserve_case=True, reduce_len=False)

print("\nSo sánh trên bài thô P3 và P1:")
for idx in (2, 0):
    print(f"\n  Raw       : {posts[idx]}")
    print(f"  whitespace: {posts[idx].split()}")
    print(f"  word_tokenize: {word_tokenize(posts[idx])}")
    print(f"  TweetTokenizer: {tweet_tok.tokenize(posts[idx])}")

# Nhận xét:
#   - whitespace: dấu câu dính vào từ ("muchhh!!!", "tomorrow...") nên cùng một
#     từ thành nhiều token khác nhau.
#   - word_tokenize: vỡ cả mention ("@", "JohnDoe") lẫn URL ("http", ":",
#     "//bit.ly/123"), hashtag thành "#", "blessed".
#   - TweetTokenizer: giữ "@JohnDoe", "#blessed", URL và emoji thành 1 token
#     (nên xóa nhiễu trước vẫn là cách an toàn). Nó vẫn có điểm yếu: dấu nháy
#     cong ’ làm "Can’t" vỡ thành "Can", "’", "t" - đó là lý do bước 1 đã
#     chuẩn hóa ’ -> ' trước. Ngoài ra "iPhone14" bị tách thành "iPhone", "14".

# Pipeline: tách trên văn bản ĐÃ xóa nhiễu, rồi bỏ các token chỉ là dấu câu.
WORD_RE = re.compile(r"[A-Za-z0-9']+")   # chữ, số (vd "iphone14", "4"), nháy


def tokenize(text: str):
    tokens = []
    for tok in tweet_tok.tokenize(text):
        if WORD_RE.fullmatch(tok) or is_emoji(tok):
            tokens.append(tok)           # giữ từ và emoji, bỏ dấu câu
    return tokens


tokenized = [tokenize(t) for t in no_noise]
show("tokens (TweetTokenizer, bỏ dấu câu, giữ emoji)", tokenized)


# ---------------------------------------------------------------------------
# 3. NORMALIZATION
# ---------------------------------------------------------------------------
# Mục tiêu: nhiều cách viết khác nhau của CÙNG một từ -> MỘT dạng chuẩn, để
# đếm chung và giảm số chiều từ vựng ("sooo" và "so" không còn là 2 từ).
banner("3. NORMALIZATION")

# 3a. Chữ thường: "OMG", "Omg", "omg" -> "omg". Emoji không bị ảnh hưởng.
lowered = [[t.lower() for t in tokens] for tokens in tokenized]
show("3a. lowercase", lowered)

# 3b. Ký tự lặp: "looooveee" -> "love", "sooo" -> "so", "muchhh" -> "much".
# Không thể đơn giản gộp mọi ký tự lặp về 1 vì từ đúng có thể chứa chữ đôi
# ("see", "need", "tomorrow"). Cách làm ở đây:
#   - chỉ đụng tới chuỗi lặp >= 3 lần (từ thật hầu như không có 3 chữ liền)
#   - thử rút mỗi chuỗi lặp còn 1 hoặc 2 ký tự, ưu tiên bản NGẮN nhất
#   - chọn bản đầu tiên là từ có trong từ điển WordNet
#   - nếu không bản nào có trong từ điển, rút còn 2 ký tự (giữ thông tin nhấn)
# Đây chỉ là heuristic; thư viện thật dùng spell-checker hoặc mô hình ngôn ngữ.
REPEAT_RE = re.compile(r"(.)\1{2,}")


def fix_elongation(word: str) -> str:
    runs = [(m.start(), m.end()) for m in REPEAT_RE.finditer(word)]
    if not runs:
        return word
    candidates = []
    for sizes in itertools.product((1, 2), repeat=len(runs)):
        out, last = [], 0
        for (s, e), n in zip(runs, sizes):
            out.append(word[last:s] + word[s] * n)
            last = e
        out.append(word[last:])
        candidates.append("".join(out))
    for cand in sorted(candidates, key=len):      # ưu tiên bản ngắn nhất
        if wordnet.synsets(cand):
            return cand
    return REPEAT_RE.sub(r"\1\1", word)           # dự phòng: rút còn 2 ký tự


deelongated = [[fix_elongation(t) if t.isalpha() else t for t in tokens]
               for tokens in lowered]
show("3b. rút ký tự lặp", deelongated)

# 3c. Mở rộng viết tắt / tiếng lóng bằng từ điển tra cứu.
# Một từ có thể mở thành NHIỀU từ ("rn" -> "right now").
# Cảnh báo nhập nhằng: "4" và "2" ở đây là "for"/"to" nhưng "4" cũng có thể là
# con số thật; "r" có thể là chữ cái R. Từ điển đơn giản không xét ngữ cảnh -
# hệ thống thật cần mô hình ngữ cảnh để quyết định.
SLANG = {
    "u": ["you"], "r": ["are"], "thx": ["thanks"], "4": ["for"], "2": ["to"],
    "rn": ["right", "now"], "omg": ["oh", "my", "god"],
    "can't": ["can", "not"],       # co từ: tách để giữ phủ định "not" riêng
}


def expand_slang(tokens):
    out = []
    for t in tokens:
        out.extend(SLANG.get(t, [t]))
    return out


normalized = [expand_slang(tokens) for tokens in deelongated]
show("3c. mở rộng viết tắt / tiếng lóng", normalized)


# ---------------------------------------------------------------------------
# 4. STOPWORD HANDLING
# ---------------------------------------------------------------------------
# Dùng danh sách stopword tiếng Anh của NLTK (198 từ).
#
# THẢO LUẬN: loại stopword có LUÔN tốt cho sentiment analysis? -> KHÔNG.
#   - Phủ định là stopword trong nhiều danh sách ("not", "no", "nor",
#     "don't"...). Xóa đi thì "not good" thành "good", đảo ngược cảm xúc!
#   - Từ nhấn mạnh như "so", "very", "too" thay đổi cường độ cảm xúc
#     ("so tired" khác "tired").
#   - Các mô hình hiện đại (BERT, GPT) dùng chính stopword để hiểu cú pháp và
#     ngữ cảnh, nên thường KHÔNG loại stopword.
#   - Còn với TF-IDF / Bag-of-Words + mô hình đơn giản thì loại stopword vẫn
#     hữu ích: giảm số chiều và giảm nhiễu.
# Cách làm ở đây: loại stopword NHƯNG giữ lại các từ phủ định.
NEGATIONS = {"no", "nor", "not", "never"} | {
    w for w in stopwords.words("english") if w.endswith("n't")}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS

banner("4. STOPWORD HANDLING")
print(f"NLTK có {len(stopwords.words('english'))} stopword; "
      f"giữ lại {len(NEGATIONS & set(stopwords.words('english')))} từ phủ định.")

# Gán nhãn từ loại (POS) trên câu ĐẦY ĐỦ (trước khi loại stopword) để lemma
# chính xác hơn - bước 5 cần. Emoji được gán nhãn tạm, ta không lemmatize chúng.
tagged = [nltk.pos_tag(tokens) for tokens in normalized]

no_stop_tagged = [[(t, tag) for t, tag in post if t not in STOPWORDS]
                  for post in tagged]
show("loại stopword (giữ phủ định)", [[t for t, _ in p] for p in no_stop_tagged])

# So sánh: nếu loại cả từ phủ định, P3 "can not wait" mất chữ "not".
naive = [[t for t in post if t not in set(stopwords.words("english"))]
         for post in normalized]
print(f"\nSo sánh P3 nếu xóa luôn cả 'not': {naive[2]}  "
      f"<- mất phủ định!\nP3 giữ phủ định          : "
      f"{[t for t, _ in no_stop_tagged[2]]}")


# ---------------------------------------------------------------------------
# 5. STEMMING vs LEMMATIZATION
# ---------------------------------------------------------------------------
# Áp dụng riêng từng phương pháp rồi so sánh.
#   Stemming (Porter)     : cắt hậu tố theo luật; nhanh, thô, kết quả có thể
#                           không phải từ thật: tired -> "tire", always ->
#                           "alway", released -> "releas".
#   Lemmatization (WordNet): tra từ điển + từ loại; trả về từ thật:
#                           tired (tính từ) -> "tired"  (đúng như đề: slide
#                           so sánh "tired" -> "tire" (stem) vs "tired" (lemma)).
banner("5. STEMMING vs LEMMATIZATION")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def to_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def stem(token: str) -> str:
    return token if is_emoji(token) else stemmer.stem(token)


def lemma(token: str, tag: str) -> str:
    return token if is_emoji(token) else lemmatizer.lemmatize(
        token, to_wordnet_pos(tag))


stemmed = [[stem(t) for t, _ in post] for post in no_stop_tagged]
lemmatized = [[lemma(t, tag) for t, tag in post] for post in no_stop_tagged]
show("stemming (Porter)", stemmed)
show("lemmatization (WordNet + POS)", lemmatized)

print("\nBảng so sánh các từ khác nhau giữa hai phương pháp:")
print(f"  {'token':<12}{'stem':<12}{'lemma':<12}")
for post in no_stop_tagged:
    for t, tag in post:
        s, l = stem(t), lemma(t, tag)
        if s != l:
            print(f"  {t:<12}{s:<12}{l:<12}")

# Kết luận: stem gọn hơn nhưng khó đọc ("releas", "alway", "tire" làm mất
# nghĩa "mệt"); lemma giữ từ thật nên dễ giải thích hơn, đổi lại chậm hơn và
# phụ thuộc chất lượng POS tag. Với sentiment, "tired" ≠ "tire" nên lemma an
# toàn hơn; với tìm kiếm / BoW lớn, stemming nhanh vẫn đủ dùng.


# ---------------------------------------------------------------------------
# 6. FINAL CLEANED DATA
# ---------------------------------------------------------------------------
# Chọn kết quả LEMMATIZATION làm dữ liệu cuối: giữ từ thật (dễ debug, hợp với
# embedding/từ điển) và tránh các lỗi cắt sai như "tired" -> "tire".
# Mỗi bài đăng là danh sách token sạch: không URL/mention/hashtag, chữ thường,
# đã sửa ký tự lặp và tiếng lóng, bỏ stopword (giữ phủ định), giữ emoji.
banner("6. FINAL CLEANED DATA")
final_tokens = lemmatized
for i, (raw, toks) in enumerate(zip(posts, final_tokens), 1):
    print(f"\nP{i} raw  : {raw}")
    print(f"P{i} clean: {toks}")

# Nếu mô hình cần chuỗi (vd TfidfVectorizer của sklearn), nối token bằng dấu cách:
final_texts = [" ".join(toks) for toks in final_tokens]
show("dạng chuỗi (đưa vào TF-IDF / sklearn)", final_texts)
