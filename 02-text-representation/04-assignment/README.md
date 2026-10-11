# Word Embeddings for Sentiment Classification

So sánh **Word2Vec CBOW**, **GloVe** và **fastText CBOW** (cùng một LSTM một chiều, một tầng) trên
bài toán phân loại cảm xúc review phim IMDB, kèm một điều kiện đối chứng embedding ngẫu nhiên.
Đây là project **độc lập** (có `requirements.txt` và `.venv` riêng, không dùng chung với repo gốc).

Báo cáo kỹ thuật (trả lời 6 câu hỏi của đề): [REPORT.md](REPORT.md) ·
Bảng "Required experiments" dạng PDF: [report/report.pdf](report/report.pdf)

## Cài đặt (môi trường sạch)

Yêu cầu Python >= 3.10 (đã chạy với 3.12.4).

```bash
cd 02-text-representation/04-assignment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Chạy

```bash
.venv/bin/python run_experiments.py            # đầy đủ: 7 điều kiện x 3 seed (~50 phút trên M1 Max/MPS)
.venv/bin/python run_experiments.py --seeds 0  # pilot 1 seed
.venv/bin/python run_experiments.py --quick    # smoke test vài phút (dữ liệu nhỏ, ghi ra results_quick/)
.venv/bin/python make_report.py --results results --out report/report.pdf   # bảng PDF (cần Google Chrome)
```

Chương trình chạy lại được: run nào đã có trong `results/runs/` sẽ được bỏ qua. Lần đầu sẽ tải IMDB
(Hugging Face) và GloVe (~128 MB, gensim-data).

## Cấu trúc

```
run_experiments.py   # điều phối: dữ liệu -> embedding -> E0..E4 x seeds -> tổng hợp
make_report.py       # sinh bảng PDF từ results*/
src/config.py        # mọi siêu tham số (một chỗ duy nhất => fairness rule)
src/data.py          # tải IMDB, tách 80/20, tokenizer, vocabulary, pad/cắt
src/embeddings.py    # Word2Vec, GloVe, fastText, căn chỉnh ma trận, coverage
src/model.py         # SentimentLSTM (dùng lại cho mọi điều kiện)
src/train.py         # vòng lặp huấn luyện, early stopping
src/evaluate.py      # chỉ số, biểu đồ, phân tích lỗi, tra cứu fastText động
results/             # đầu ra (xem bên dưới)
```

## Đầu ra trong `results/`

| File | Nội dung |
|---|---|
| `metrics.csv` | mọi run: thí nghiệm, embedding, trainable/frozen, seed, accuracy/precision/recall/F1/macro-F1/ROC-AUC (test chính thức), chỉ số trên tập con từ hiếm, thời gian, số tham số, RSS đỉnh. *(Đây là "results.csv" của đề.)* |
| `summary.csv` | trung bình ± độ lệch chuẩn theo điều kiện |
| `coverage.csv` | coverage từ vựng/token của từng nguồn embedding và tỉ lệ `<unk>` val/test |
| `learning_curves.png`, `learning_curves_trainable.png` | loss train/validation |
| `confusion_matrices.png`, `confusion_matrices_trainable.png` | confusion matrix (seed 0) trên test |
| `error_analysis.md` | 20 review bị cả E0-E3 đoán sai |
| `subword_demo.txt` | minh họa vector fastText cho từ lạ / sai chính tả |
| `split_indices.json`, `rare_subset_indices.json` | chỉ số chia train/val và tập con từ hiếm (tái lập) |
| `runs/` | json + xác suất dự đoán của từng run |

## Thiết lập thí nghiệm

- **Dữ liệu:** IMDB 25.000 train / 25.000 test. Train chia 80/20 (20.000 / 5.000) bằng hoán vị seed 42; test chính thức chỉ dùng ở bước đánh giá cuối.
- **Tokenizer:** chữ thường, giữ chữ cái và dấu nháy `'`, bỏ thẻ `<br />`. Vocabulary chỉ từ review train, từ xuất hiện < 2 lần → `<unk>` (46.321 mục gồm `<pad>`, `<unk>`). Cắt/pad về 300 token.
- **Word2Vec / fastText:** CBOW, dim 100, window 5, min_count 2, negative sampling (5), 10 epoch; fastText n-gram 3..6. Chỉ huấn luyện trên token review train (không nhãn, không val/test). Huấn luyện một lần (seed 42), dùng cho mọi seed của LSTM.
- **GloVe:** `glove-wiki-gigaword-100` từ gensim-data (Pennington et al. 2014; Wikipedia 2014 + Gigaword 5, 6 tỷ token, chữ thường). Corpus **khác** IMDB.
- **LSTM:** 1 tầng một chiều, hidden 128, 1 logit, `BCEWithLogitsLoss`, Adam lr 1e-3 (hằng số), batch 64, early stopping theo val loss (patience 2, tối đa 20 epoch), clip gradient 5. Hàng thiếu của Word2Vec/GloVe ~ N(0, 0.1²); hàng `<pad>` = 0.
- **Thí nghiệm:** E0 random (frozen), E1 Word2Vec, E2 GloVe, E3 fastText (frozen), E4 cả ba (trainable); mỗi cái 3 seed (0, 1, 2).
- **Mở rộng đã làm:** tra cứu fastText động (`E3dyn`: `wv.get_vector` cho từ ngoài vocabulary thay cho `<unk>`); tập con "từ hiếm" (10% review test có tỉ lệ `<unk>` cao nhất).

## Khác biệt / lưu ý so với đề

- `load_dataset("imdb")` lỗi với `huggingface_hub` mới, nên code thử `"imdb"` rồi chuyển sang `"stanfordnlp/imdb"` (cùng dataset).
- Không dùng `pack_padded_sequence`: padding nằm cuối chuỗi và LSTM chạy xuôi, nên lấy đầu ra tại vị trí `length-1` tương đương và chạy được trên MPS.
- "RSS đỉnh" là của cả tiến trình (tích lũy, gồm cả 3 mô hình embedding trong bộ nhớ), không phải riêng từng run.
- Chạy trên GPU Apple (MPS) nên không bit-for-bit tái lập giữa các máy; seed vẫn cố định.
- Chưa làm: tập sai chính tả có kiểm soát, Skip-gram, BiLSTM, attention, GloVe tự huấn luyện (phần mở rộng tùy chọn).

## Phần cứng

Apple M1 Max, 32 GB RAM, macOS 26.7, Python 3.12.4, PyTorch 2.14.1 (MPS), gensim 4.4.0.

## Nguồn ngoài và công cụ sinh mã

- IMDB Large Movie Review Dataset (Maas et al. 2011) qua Hugging Face Datasets (`stanfordnlp/imdb`).
- GloVe `glove-wiki-gigaword-100` qua `gensim.downloader`.
- Thư viện: PyTorch, Gensim, Hugging Face Datasets, scikit-learn, Matplotlib.
- Mã nguồn và bản nháp báo cáo được viết với sự hỗ trợ của công cụ sinh mã Claude Code (Anthropic); kết quả đều do chương trình tạo ra từ các lần chạy trên.
