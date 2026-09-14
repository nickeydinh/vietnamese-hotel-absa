# Phân tích Cảm xúc theo Khía cạnh cho Đánh giá Khách sạn Tiếng Việt

Hệ thống Aspect-Based Sentiment Analysis (ABSA) cho review khách sạn tiếng Việt, so sánh hai mô hình ngôn ngữ tiền huấn luyện **PhoBERT** và **ViSoBERT**.

> Đồ án cơ sở — Ngành Khoa học Dữ liệu, Đại học Công nghệ TP.HCM (HUTECH), 2026
> Thực hiện: Đinh Quốc Khánh · GVHD: Lê Cung Tưởng

---

## Bài toán

Cho một câu đánh giá khách sạn bằng tiếng Việt, hệ thống xác định **6 khía cạnh** và gán cho mỗi khía cạnh một trong **5 trạng thái cảm xúc**.

| | Giá trị |
|---|---|
| **Khía cạnh (aspect)** | HOTEL, LOCATION, ROOMS, FACILITIES, FOOD&DRINKS, SERVICE |
| **Cảm xúc (sentiment)** | NONE, POSITIVE, NEGATIVE, NEUTRAL, CONFLICT |

Ví dụ:

```
Input:  "Phòng sạch nhưng nhân viên phục vụ chậm"
Output: ROOMS: POSITIVE
        SERVICE: NEGATIVE
```

`NONE` nghĩa là khía cạnh không được nhắc tới. `CONFLICT` dùng khi cùng một khía cạnh vừa có nhận xét tích cực vừa có tiêu cực.

---

## Kết quả

Đánh giá trên tập test độc lập:

| Mô hình | Mean Aspect Accuracy | Macro F1 | Weighted F1 | Exact Match |
|---|---|---|---|---|
| **PhoBERT** | **0.9295** | **0.6517** | **0.9322** | **0.7096** |
| ViSoBERT | 0.9086 | 0.5911 | 0.9106 | 0.6289 |

PhoBERT vượt ViSoBERT ở cả bốn chỉ số, rõ rệt nhất ở Exact Match (+0.081) và Macro F1 (+0.061) — hai chỉ số phản ánh sát nhất hiệu quả thực tế khi dữ liệu mất cân bằng và cần dự đoán đồng thời nhiều khía cạnh. PhoBERT được chọn làm mô hình chính cho pipeline suy luận.

### Macro F1 theo từng khía cạnh

| Aspect | PhoBERT | ViSoBERT |
|---|---|---|
| HOTEL | 0.7285 | 0.6585 |
| LOCATION | 0.5120 | **0.5339** |
| ROOMS | 0.7051 | 0.6380 |
| FACILITIES | 0.5157 | 0.4096 |
| FOOD&DRINKS | **0.7575** | 0.6632 |
| SERVICE | 0.6914 | 0.6431 |

PhoBERT thắng ở 5/6 khía cạnh. `LOCATION` và `FACILITIES` có Macro F1 thấp do số mẫu thực sự xuất hiện quá ít (xem phần Hạn chế).

---

## Dữ liệu

Dữ liệu được tự thu thập từ Agoda (khách sạn tại TP.HCM) và **gán nhãn thủ công**.

| | Số lượng |
|---|---|
| Đơn vị văn bản ban đầu | 8.793 |
| Sau khi khử trùng lặp | 7.595 |
| Số review gốc | 5.715 |
| Tổng số nhãn sentiment (không tính NONE) | 9.338 |

### Phân bố nhãn

| Sentiment | Số lượng |
|---|---|
| POSITIVE | 5.233 |
| NEGATIVE | 2.913 |
| CONFLICT | 637 |
| NEUTRAL | 555 |

Dữ liệu mất cân bằng ở cả hai chiều: `NEUTRAL` và `CONFLICT` chỉ chiếm khoảng 13% số nhãn; khía cạnh `HOTEL` xuất hiện trong 63,4% đơn vị văn bản trong khi `LOCATION` chỉ 7,9%.

### Chia tập dữ liệu

| Tập | Số review |
|---|---|
| Train | 4.558 |
| Validation | 569 |
| Test | 588 |

Việc chia được thực hiện **ở cấp review, không phải cấp câu**. Lý do: nhiều đơn vị văn bản xuất phát từ cùng một review gốc và chia sẻ ngữ cảnh, cách diễn đạt — nếu chia ngẫu nhiên theo câu thì một phần review sẽ nằm ở train, phần còn lại ở test, gây **data leakage** và làm kết quả đánh giá bị thổi phồng.

Trên nền đó, mỗi review được biểu diễn bằng một label profile nhị phân dạng `aspect_sentiment`, rồi dùng `MultilabelStratifiedShuffleSplit` để giữ phân bố tổ hợp nhãn ổn định giữa ba tập.

---

## Phương pháp

### Tiền xử lý tiếng Việt

1. Trích xuất và lưu tạm các cụm từ đã được nối sẵn bằng dấu gạch dưới (ví dụ `bữa_sáng`)
2. Tách từ bằng **VnCoreNLP** (annotator `wseg`)
3. Khôi phục lại các cụm đã lưu, tránh để VnCoreNLP phân tách nhầm từ ghép đã chuẩn hóa thủ công

### Kiến trúc mô hình

Multi-output classification: một encoder tiền huấn luyện dùng chung, phía trên gắn **6 head phân loại độc lập**, mỗi head cho một khía cạnh với 5 lớp đầu ra.

```
Input text
    │
    ▼
[PhoBERT / ViSoBERT encoder]
    │
    ├──► Head HOTEL        (5 lớp)
    ├──► Head LOCATION     (5 lớp)
    ├──► Head ROOMS        (5 lớp)
    ├──► Head FACILITIES   (5 lớp)
    ├──► Head FOOD&DRINKS  (5 lớp)
    └──► Head SERVICE      (5 lớp)
```

Độ dài chuỗi: `MAX_LENGTH = 64` cho PhoBERT, `128` cho ViSoBERT.

### Xử lý mất cân bằng

Cross Entropy Loss với **class weights tính riêng cho từng khía cạnh**; loss tổng là trung bình qua 6 khía cạnh. Trọng số tối đa (`max_class_weight`) được đưa vào không gian tìm kiếm siêu tham số thay vì cố định.

### Tối ưu siêu tham số

Dùng **Optuna**, hàm mục tiêu dựa trên Mean Aspect Macro F1 trên tập validation.

| Siêu tham số | PhoBERT | ViSoBERT |
|---|---|---|
| Learning rate | 1.7566 × 10⁻⁵ | 1.7757 × 10⁻⁵ |
| Batch size | 16 | 16 |
| Weight decay | 0.01 | 0.10 |
| Warmup ratio | 0.10 | 0.05 |
| Dropout | 0.20 | 0.30 |
| Max class weight | 5.0 | 3.0 |
| Số trial | 8 | 6 |

---

## Cài đặt

```bash
git clone https://github.com/<username>/vietnamese-hotel-absa.git
cd vietnamese-hotel-absa
pip install -r requirements.txt
```

VnCoreNLP cần Java 8 trở lên và file model riêng:

```bash
# Tải VnCoreNLP model
python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='./vncorenlp')"
```

### Model weights

Trọng số mô hình đã fine-tune không được đưa lên repo do giới hạn dung lượng của GitHub. Tải tại: **[link Google Drive]**

Giải nén vào thư mục `models/`.

---

## Sử dụng

### Chạy suy luận

```python
from src.inference import ABSAPredictor

predictor = ABSAPredictor(model_path="models/phobert")
result = predictor.predict("Vị trí thuận tiện, gần trung tâm nhưng phòng hơi nhỏ")

# LOCATION: POSITIVE
# ROOMS:    NEGATIVE
```

### Chạy giao diện demo

```bash
streamlit run app.py
```

Giao diện cho phép nhập review bất kỳ, chọn mô hình suy luận (PhoBERT hoặc ViSoBERT) và xem kết quả aspect–sentiment cho từng đơn vị văn bản trong review.

---

## Cấu trúc thư mục

```
vietnamese-hotel-absa/
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Làm sạch, khử trùng lặp
│   ├── 02_eda.ipynb                # Phân tích phân bố nhãn
│   ├── 03_preprocessing.ipynb      # VnCoreNLP, chia tập
│   ├── 04_hpo_training.ipynb       # Optuna + fine-tuning
│   └── 05_evaluation.ipynb         # Đánh giá, confusion matrix
├── src/
│   ├── model.py                    # Kiến trúc multi-head
│   ├── dataset.py                  # Dataset, tokenization
│   ├── train.py                    # Vòng lặp huấn luyện
│   ├── evaluate.py                 # Các chỉ số đánh giá
│   └── inference.py                # Pipeline suy luận
├── app.py                          # Giao diện demo
├── requirements.txt
└── README.md
```

---

## Hạn chế

Đây là những điểm yếu đã được xác định, không phải danh sách tính năng còn thiếu:

- **Mất cân bằng nhãn.** `NEUTRAL` và `CONFLICT` có quá ít mẫu; class weights chỉ giảm thiểu được một phần. Macro F1 (0.65) thấp hơn đáng kể so với Weighted F1 (0.93) là bằng chứng trực tiếp.
- **Gán nhãn bởi một người.** Không có kiểm tra chéo giữa nhiều người gán nhãn. Ranh giới giữa `NEUTRAL` và `NEGATIVE` đôi khi mờ, nên tính nhất quán của nhãn phụ thuộc vào cảm nhận chủ quan.
- **Phạm vi dữ liệu hẹp.** Toàn bộ review lấy từ Agoda, chỉ ở khách sạn tại TP.HCM. Chưa rõ mô hình hoạt động ra sao với resort, homestay, hostel hoặc nền tảng khác.
- **Thiếu baseline truyền thống.** Chỉ so sánh hai mô hình pretrained với nhau, chưa có baseline như TF-IDF + SVM để đo mức cải thiện thực sự.

## Hướng phát triển

- Mở rộng dữ liệu về cả quy mô lẫn đa dạng nguồn
- Bổ sung baseline truyền thống để định lượng đóng góp của mô hình pretrained
- Thử các kỹ thuật xử lý mất cân bằng khác (Focal Loss, oversampling ở cấp nhãn hiếm)
- Kiểm tra chéo nhãn với nhiều người gán để đo độ tin cậy (inter-annotator agreement)

---

## Tài liệu tham khảo

- Nguyen & Nguyen (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP.
- Nguyen et al. (2023). *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing.* EMNLP.
- Vu et al. (2018). *VnCoreNLP: A Vietnamese Natural Language Processing Toolkit.* NAACL.
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
