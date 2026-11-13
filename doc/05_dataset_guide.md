# 05 – Hướng Dẫn Dataset

---

## 1. Tổng Quan Datasets Sử Dụng

| Dataset | Tên đầy đủ | J (streams) | D (attributes) | Sparsity | Nhiệm vụ |
|---------|-----------|-------------|----------------|----------|---------|
| **ICU** | PhysioNet Challenge 2012 | 11,988 | 37 | ≈ 80% | Imputation + Prediction |
| **Airquality** | Multi-site Air Quality Monitoring | — | 11 | — | Imputation |

---

## 2. Dataset ICU (PhysioNet Challenge 2012)

### 2.1 Giới Thiệu

**PhysioNet Challenge 2012** là dataset chuẩn trong cộng đồng nghiên cứu y tế và machine learning. Dataset này chứa hồ sơ y tế của bệnh nhân trong đơn vị chăm sóc đặc biệt (ICU) trong 48 giờ đầu tiên nhập viện.

- **Nguồn**: [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
- **Paper gốc**: Silva et al., "Predicting In-hospital Mortality of ICU Patients", 2012.
- **Số lượng bệnh nhân**: 11,988 (chia 3 tập: Set A, B, C)
- **Thời gian theo dõi**: 48 giờ từ khi nhập ICU
- **Tần suất đo**: Irregular (mỗi lần lấy máu hoặc kiểm tra)

### 2.2 Các Thuộc Tính (D=37)

| # | Tên thuộc tính | Đơn vị | Mô tả |
|---|---------------|--------|-------|
| 1 | Albumin | g/dL | Protein albumin trong máu |
| 2 | ALP | IU/L | Alkaline phosphatase |
| 3 | ALT | IU/L | Alanine aminotransferase (men gan) |
| 4 | AST | IU/L | Aspartate aminotransferase (men gan) |
| 5 | Bilirubin | mg/dL | Bilirubin toàn phần |
| 6 | BUN | mg/dL | Blood urea nitrogen (ure máu) |
| 7 | Cholesterol | mg/dL | Cholesterol |
| 8 | Creatinine | mg/dL | Creatinine (chức năng thận) |
| 9 | DiasABP | mmHg | Huyết áp tâm trương (xâm lấn) |
| 10 | FiO2 | — | Nồng độ oxy trong khí thở |
| 11 | GCS | — | Glasgow Coma Scale (ý thức) |
| 12 | Glucose | mg/dL | Đường huyết |
| 13 | HCO3 | mEq/L | Bicarbonate |
| 14 | HCT | % | Hematocrit |
| 15 | HR | bpm | Nhịp tim |
| 16 | K | mEq/L | Kali huyết thanh |
| 17 | Lactate | mmol/L | Lactate (chỉ số thiếu oxy mô) |
| 18 | Mg | mEq/L | Magie huyết thanh |
| 19 | MAP | mmHg | Huyết áp trung bình |
| 20 | MechVent | — | Thở máy cơ học (0/1) |
| 21 | Na | mEq/L | Natri huyết thanh |
| 22 | NIDiasABP | mmHg | Huyết áp tâm trương (không xâm lấn) |
| 23 | NIMAP | mmHg | MAP không xâm lấn |
| 24 | NISysABP | mmHg | Huyết áp tâm thu không xâm lấn |
| 25 | PaCO2 | mmHg | CO2 động mạch |
| 26 | PaO2 | mmHg | O2 động mạch |
| 27 | pH | — | Độ pH máu |
| 28 | Platelets | 1000/μL | Tiểu cầu |
| 29 | RespRate | breaths/min | Nhịp thở |
| 30 | SaO2 | % | Độ bão hòa O2 (xâm lấn) |
| 31 | SysABP | mmHg | Huyết áp tâm thu (xâm lấn) |
| 32 | Temp | °C | Nhiệt độ cơ thể |
| 33 | TroponinI | μg/L | Troponin I (tim) |
| 34 | TroponinT | μg/L | Troponin T (tim) |
| 35 | Urine | mL | Lượng nước tiểu |
| 36 | WBC | 1000/μL | Bạch cầu |
| 37 | Weight | kg | Cân nặng |

### 2.3 Tại Sao Sparsity Cao Đến 80%?

Lý do chính:
1. **Không phải lúc nào cũng lấy máu xét nghiệm**: Nhiều xét nghiệm (ALT, AST, Creatinine...) chỉ được làm 1-2 lần/ngày.
2. **Khác nhau theo ca trực**: Mỗi ca trực có thể đo các chỉ số khác nhau.
3. **Thiếu thiết bị**: Một số ICU không có đủ thiết bị đo tất cả chỉ số.
4. **Event-driven**: Một số chỉ số chỉ được đo khi có nghi ngờ lâm sàng.

### 2.4 Label (Task 2)

**In-hospital mortality**: Bệnh nhân có tử vong trong quá trình nằm viện không?
- `0`: Sống sót
- `1`: Tử vong

Tỷ lệ label: Khoảng 14% tử vong (imbalanced dataset).

### 2.5 Cách Tải Dataset

```bash
# Bước 1: Đăng ký tài khoản PhysioNet tại https://physionet.org
# Bước 2: Hoàn thành CITI training (bắt buộc, khoảng 1-2 giờ)
# Bước 3: Đăng nhập và tải về

# Dùng wget (Linux/Mac):
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
    https://physionet.org/files/challenge-2012/1.0.0/

# Hoặc dùng script:
python scripts/download_data.py --dataset icu --physionet_user YOUR_USERNAME
```

**Cấu trúc file sau tải:**
```
data/raw/icu/
├── set-a/          # 4,000 patients
│   ├── 132539.txt
│   ├── 132540.txt
│   └── ...
├── set-b/          # 4,000 patients
├── set-c/          # 4,000 patients (without labels)
├── Outcomes-a.txt  # Labels for set-a
└── Outcomes-b.txt  # Labels for set-b
```

**Format file bệnh nhân** (ví dụ `132539.txt`):
```
Time,Parameter,Value
00:07,RecordID,132539
00:07,Age,54
00:07,Gender,0
00:07,Height,-1
00:07,ICUType,4
00:07,Weight,75.6
01:19,GCS,15
01:36,HR,88
...
```

---

## 3. Dataset Airquality

### 3.1 Giới Thiệu

Dataset quan trắc chất lượng không khí từ nhiều trạm đo. Mỗi trạm đo nhiều chất ô nhiễm và điều kiện thời tiết.

- **Loại dữ liệu**: Time series đa thuộc tính từ nhiều trạm quan trắc
- **Thuộc tính**: D=11

### 3.2 Các Thuộc Tính (D=11)

| # | Tên | Đơn vị | Mô tả |
|---|-----|--------|-------|
| 1 | PM2.5 | μg/m³ | Hạt bụi mịn ≤2.5μm (nguy hiểm nhất) |
| 2 | PM10 | μg/m³ | Hạt bụi thô ≤10μm |
| 3 | SO2 | μg/m³ | Lưu huỳnh dioxide |
| 4 | NO2 | μg/m³ | Nitrogen dioxide |
| 5 | CO | mg/m³ | Carbon monoxide |
| 6 | O3 | μg/m³ | Ozone |
| 7 | Temp | °C | Nhiệt độ không khí |
| 8 | Pres | hPa | Áp suất khí quyển |
| 9 | Dewp | °C | Điểm sương |
| 10 | Rain | mm | Lượng mưa |
| 11 | WS | m/s | Tốc độ gió |

### 3.3 Cách Tải Dataset

```bash
# Option 1: Từ paper MPIN (nếu có link)
# Xem GitHub: https://github.com/XLI-2020/MPIN

# Option 2: UCI Air Quality Dataset
# https://archive.ics.uci.edu/ml/datasets/Air+Quality
python scripts/download_data.py --dataset airquality

# Option 3: Tải thủ công và đặt vào:
# data/raw/airquality/
```

---

## 4. Tiền Xử Lý Dữ Liệu

### 4.1 ICU Data Pipeline

```python
# Bước 1: Parse raw .txt files
# Mỗi file bệnh nhân → pivot table: rows=time_buckets, cols=37 attributes

# Bước 2: Tạo time buckets (1 giờ)
# Nếu có nhiều đo trong 1 giờ → lấy giá trị cuối
# → Tensor: (11988, 48, 37) với 48 giờ

# Bước 3: Normalize
# Min-Max normalize mỗi thuộc tính: x_norm = (x - x_min) / (x_max - x_min)
# Chỉ dùng mean/std của tập train để tránh data leakage

# Bước 4: Tạo mask M
# M[j,t,d] = 1 nếu X[j,t,d] không phải NaN
# M[j,t,d] = 0 nếu X[j,t,d] là NaN

# Bước 5: Tumbling windows
# Window size T (ví dụ T=12 → 12 giờ/chunk)
```

### 4.2 Airquality Data Pipeline

```python
# Bước 1: Load CSV files
# Bước 2: Resample theo giờ (nếu cần)
# Bước 3: Normalize per attribute
# Bước 4: Tạo mask M
# Bước 5: Tumbling windows
```

### 4.3 Normalization

**Min-Max Normalization** (đưa về [0, 1]):

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Lý do chọn Min-Max:
- Giữ phân phối gốc (không giả định Gaussian như Z-score).
- Phù hợp với Euclidean distance trong similarity graph.
- Các thuộc tính có cùng scale → khoảng cách công bằng hơn.

---

## 5. Vị Trí File Trong Dự Án

```
BTL/
├── data/
│   ├── raw/
│   │   ├── icu/
│   │   │   ├── set-a/           # Raw patient records
│   │   │   ├── set-b/
│   │   │   ├── Outcomes-a.txt   # Mortality labels
│   │   │   └── Outcomes-b.txt
│   │   └── airquality/
│   │       └── airquality.csv   # Raw measurements
│   └── processed/
│       ├── icu/
│       │   ├── X_icu.npy        # Data tensor (J, T, D)
│       │   ├── M_icu.npy        # Mask tensor (J, T, D)
│       │   └── y_icu.npy        # Labels (J,) for prediction
│       └── airquality/
│           ├── X_airquality.npy
│           └── M_airquality.npy
```

---

## 6. Thống Kê Nhanh Dataset (Placeholder)

> Sẽ được điền sau khi chạy `notebooks/01_data_exploration.ipynb`

### ICU

| Thống kê | Giá trị |
|---------|---------|
| Số bệnh nhân (J) | 11,988 |
| Số timesteps (T) | 48 (giờ) |
| Số thuộc tính (D) | 37 |
| Overall missing rate | ≈ 80% |
| Attribute missing rate (min) | — |
| Attribute missing rate (max) | — |
| Mortality rate | ≈ 14% |

### Airquality

| Thống kê | Giá trị |
|---------|---------|
| Số trạm (J) | — |
| Số thuộc tính (D) | 11 |
| Overall missing rate | — |

---

*Chạy `python scripts/download_data.py --help` để xem hướng dẫn tải dữ liệu chi tiết.*
