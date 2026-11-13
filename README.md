# Big Data Assignment – Multiple Stream Integration with Temporal Data
## Data Imputation & Prediction using MPIN

---

## Mô Tả Dự Án

Dự án này nghiên cứu và cài đặt mô hình **MPIN (Message Propagation Imputation Network)** được đề xuất trong bài báo:

> *"Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation"*
> – VLDB 2023, arXiv:2311.07344

Vấn đề trung tâm là **điền giá trị bị thiếu (missing value imputation)** trong các luồng dữ liệu cảm biến đa thuộc tính theo thời gian thực. Dữ liệu thiếu xảy ra rất phổ biến trong các hệ thống IoT, thiết bị y tế, và trạm quan trắc môi trường – gây ảnh hưởng nghiêm trọng đến chất lượng phân tích và quyết định.

### Hai nhiệm vụ chính:
1. **Task 1 – Imputation**: Điền giá trị bị thiếu bằng MPIN và so sánh với các baseline (MEAN, KNN, MICE, MF, BRITS, SAITS, FeaProp).
2. **Task 2 – Prediction**: Dùng dữ liệu đã điền để dự đoán downstream (phân loại tỷ lệ sống sót bệnh nhân ICU).

---

## Cấu Trúc Thư Mục

```
BTL/
├── README.md                    # File này – tổng quan dự án
├── requirements.txt             # Thư viện Python cần cài
├── .gitignore                   # Git ignore rules
│
├── configs/
│   └── config.yaml              # Toàn bộ hyperparameters và cấu hình
│
├── data/
│   ├── raw/                     # Dữ liệu gốc (chưa xử lý)
│   │   ├── icu/                 # PhysioNet ICU 2012 dataset
│   │   └── airquality/          # Air Quality dataset
│   └── processed/               # Dữ liệu sau tiền xử lý
│       ├── icu/
│       └── airquality/
│
├── doc/
│   ├── 01_project_overview.md   # Tổng quan dự án
│   ├── 02_model_selection.md    # Lý do chọn MPIN, so sánh các phương pháp
│   ├── 03_how_it_works.md       # Giải thích kỹ thuật chi tiết
│   ├── 04_project_plan.md       # Kế hoạch 7 tuần
│   └── 05_dataset_guide.md      # Hướng dẫn dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Khám phá dữ liệu
│   └── 02_mpin_demo.ipynb           # Demo MPIN trên ví dụ nhỏ
│
├── results/
│   ├── metrics/                 # Kết quả MAE/MRE các phương pháp
│   ├── plots/                   # Biểu đồ so sánh
│   └── checkpoints/             # Model checkpoints
│
├── scripts/
│   ├── download_data.py         # Script tải dữ liệu
│   ├── run_imputation.py        # Chạy pipeline imputation
│   └── run_prediction.py        # Chạy pipeline prediction
│
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py            # Load & window datasets
    │   └── preprocessor.py     # Tiền xử lý, importance score
    ├── models/
    │   ├── __init__.py
    │   ├── similarity_graph.py  # Xây dựng KNN similarity graph
    │   ├── mpin.py              # Mô hình MPIN chính
    │   ├── continuous_framework.py  # MPIN-DM: data/model update
    │   ├── baselines.py         # Các baseline imputation
    │   └── predictor.py         # Mô hình dự đoán (LSTM, Logistic)
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py           # MAE, MRE, RMSE
    │   └── compare.py           # So sánh tất cả phương pháp
    ├── utils/
    │   ├── __init__.py
    │   └── helpers.py           # Utilities (seed, logging, checkpoint)
    └── visualization/
        ├── __init__.py
        └── plots.py             # Các hàm vẽ biểu đồ
```

---

## Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu cầu môi trường
- Python 3.8+
- CUDA (khuyến nghị cho PyTorch, có thể chạy CPU)

### 2. Cài đặt thư viện

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Cài thư viện
pip install -r requirements.txt
```

### 3. Tải dữ liệu

```bash
python scripts/download_data.py --dataset icu
python scripts/download_data.py --dataset airquality
```

### 4. Chạy imputation

```bash
# Chạy MPIN trên ICU dataset
python scripts/run_imputation.py --dataset icu --method mpin --config configs/config.yaml --output results/metrics/

# Chạy tất cả baseline
python scripts/run_imputation.py --dataset icu --method all --config configs/config.yaml --output results/metrics/

# Chạy trên Airquality
python scripts/run_imputation.py --dataset airquality --method mpin --config configs/config.yaml --output results/metrics/
```

### 5. Chạy prediction (Task 2)

```bash
# Phân loại tỷ lệ sống sót ICU
python scripts/run_prediction.py --dataset icu --imputation_method mpin --predictor logistic --config configs/config.yaml
```

### 6. Khám phá dữ liệu (Notebook)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Kết Quả Nhanh (Placeholder)

> Kết quả sẽ được cập nhật sau khi chạy thực nghiệm đầy đủ.

### ICU Dataset (D=37, sparsity≈80%)

| Method | MAE ↓ | MRE ↓ |
|--------|--------|--------|
| MEAN   | —      | —      |
| KNN    | —      | —      |
| MICE   | —      | —      |
| MF     | —      | —      |
| BRITS  | —      | —      |
| SAITS  | —      | —      |
| FeaProp (FP) | — | —    |
| **MPIN** | **—** | **—** |

### Airquality Dataset (D=11)

| Method | MAE ↓ | MRE ↓ |
|--------|--------|--------|
| MEAN   | —      | —      |
| KNN    | —      | —      |
| **MPIN** | **—** | **—** |

### Task 2 – ICU Survival Prediction

| Imputation | Accuracy | AUC |
|-----------|----------|-----|
| Raw (mean fill) | — | — |
| MPIN | — | — |

---

## Tài Liệu Tham Khảo

- Li, X. et al. (2023). *Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation*. VLDB 2023. [arXiv:2311.07344](https://arxiv.org/abs/2311.07344)
- GitHub MPIN: https://github.com/XLI-2020/MPIN
- PhysioNet Challenge 2012: https://physionet.org/content/challenge-2012/1.0.0/
- Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
