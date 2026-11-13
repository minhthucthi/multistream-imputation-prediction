# 01 – Tổng Quan Dự Án

## Multiple Stream Integration with Temporal Data: Data Imputation & Prediction

---

## 1. Phát Biểu Bài Toán

Trong thực tế, các hệ thống IoT, thiết bị y tế, và trạm quan trắc môi trường liên tục thu thập dữ liệu từ nhiều cảm biến (sensor streams) theo thời gian. Mỗi cảm biến đo nhiều thuộc tính (attributes) khác nhau, tạo thành các **luồng dữ liệu đa thuộc tính (multi-attribute sensor data streams)**.

### Vấn đề: Dữ liệu thiếu (Missing Values)

Trong thực tế, một lượng lớn giá trị trong các luồng dữ liệu này bị **thiếu (missing)** do nhiều nguyên nhân:

- **Lỗi cảm biến**: Cảm biến bị hỏng hoặc mất điện tạm thời.
- **Truyền thông không ổn định**: Mất gói tin (packet loss) trên mạng không dây.
- **Giới hạn tần suất đo**: Một số thuộc tính không được đo liên tục.
- **Chính sách ghi nhận**: Chỉ ghi khi có thay đổi đáng kể (event-driven logging).

**Ví dụ cụ thể:**
- Trong bệnh viện ICU: Không phải lúc nào bác sĩ cũng lấy máu xét nghiệm đủ 37 chỉ số. Dữ liệu có thể bị thiếu đến 80%.
- Trạm quan trắc không khí: Một số cảm biến PM2.5 có thể ngoại tuyến trong vài giờ.

### Định nghĩa hình thức:

Cho một tập hợp $J$ luồng dữ liệu (streams), mỗi luồng $j$ có $D$ thuộc tính. Trong cửa sổ thời gian $T$, dữ liệu được biểu diễn bởi:

$$\mathbf{X} \in \mathbb{R}^{J \times T \times D}, \quad \mathbf{M} \in \{0, 1\}^{J \times T \times D}$$

Trong đó $M_{j,t,d} = 0$ nghĩa là giá trị $X_{j,t,d}$ bị thiếu.

**Mục tiêu**: Ước lượng các giá trị $\hat{X}_{j,t,d}$ cho tất cả vị trí bị thiếu ($M_{j,t,d} = 0$) sao cho $\hat{X}_{j,t,d} \approx X_{j,t,d}^{true}$.

---

## 2. Tại Sao Vấn Đề Này Quan Trọng?

### 2.1 Tác động trong lĩnh vực y tế

Trong bệnh viện ICU (Intensive Care Unit), các chỉ số sinh tồn của bệnh nhân (nhịp tim, huyết áp, nồng độ oxy...) được theo dõi liên tục. Nếu dữ liệu bị thiếu:
- Hệ thống cảnh báo sớm (early warning system) có thể không phát hiện nguy cơ.
- Mô hình dự đoán kết quả điều trị (mortality prediction) cho kết quả sai lệch.
- Bác sĩ đưa ra quyết định dựa trên thông tin không đầy đủ.

### 2.2 Tác động trong giám sát môi trường

Các chỉ số chất lượng không khí (PM2.5, NO2, O3...) được dùng để:
- Cảnh báo ô nhiễm cho cộng đồng.
- Lập chính sách kiểm soát phát thải.
- Nghiên cứu tác động sức khỏe dài hạn.

Dữ liệu thiếu làm giảm độ chính xác của các phân tích này.

### 2.3 Thách thức kỹ thuật đặc thù

Dữ liệu luồng cảm biến đa thuộc tính có 4 thách thức đặc biệt so với bài toán missing data thông thường (do bài báo MPIN đặt ra):

| # | Thách thức | Mô tả |
|---|-----------|-------|
| **C1** | **Aperiodicity** (Không chu kỳ) | Tần suất đo không đều, timestamps không đều nhau |
| **C2** | **Concurrency** (Đồng thời) | Nhiều luồng hoạt động song song, cần xử lý cùng lúc |
| **C3** | **Heterogeneity** (Không đồng nhất) | Các thuộc tính có đơn vị, phân phối, và tỷ lệ thiếu rất khác nhau |
| **C4** | **Sparsity** (Thưa thớt) | Tỷ lệ thiếu rất cao (lên đến 85%), dữ liệu có sẵn rất ít |

---

## 3. Tiếp Cận Hai Tầng

Dự án này chia bài toán thành hai nhiệm vụ nối tiếp nhau:

```
Raw Sensor Streams
        ↓
[ Task 1: Imputation ]
  → MPIN điền giá trị bị thiếu
  → So sánh với baselines (MAE, MRE)
        ↓
Imputed Complete Data
        ↓
[ Task 2: Prediction ]
  → Dùng dữ liệu đã điền để train classifier
  → ICU: Dự đoán tỷ lệ sống sót bệnh nhân
  → Đánh giá: Accuracy, AUC-ROC
```

### Task 1 – Imputation (Điền giá trị thiếu)

**Mục tiêu**: Điền giá trị thiếu chính xác nhất có thể.

**Phương pháp đề xuất**: MPIN – sử dụng lan truyền thông điệp (message propagation) trên đồ thị tương đồng để tổng hợp thông tin từ các stream láng giềng.

**Đánh giá**:
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình tại các vị trí thiếu.
- **MRE** (Mean Relative Error): Sai số tương đối.

### Task 2 – Downstream Prediction

**Mục tiêu**: Kiểm chứng rằng imputation tốt hơn → downstream task tốt hơn.

**Cụ thể với ICU**:
- Label: Bệnh nhân sống sót (0) hay tử vong (1) trong ICU.
- Features: 37 chỉ số sinh tồn sau khi được điền bởi MPIN.
- Classifier: Logistic Regression (đơn giản, dễ diễn giải).

---

## 4. Phạm Vi Thực Nghiệm

### Datasets

| Dataset | Số luồng (J) | Số thuộc tính (D) | Sparsity | Nguồn |
|---------|-------------|-------------------|----------|-------|
| ICU     | 11,988      | 37                | ≈ 80%    | PhysioNet Challenge 2012 |
| Airquality | —        | 11                | —        | Giám sát chất lượng không khí |

### Baselines so sánh

| Phương pháp | Loại |
|-------------|------|
| MEAN | Statistical |
| KNN | Instance-based |
| MICE | Iterative statistical |
| MF (Matrix Factorization) | Low-rank |
| BRITS | Deep learning (RNN) |
| SAITS | Deep learning (Attention) |
| FeaProp (FP) | Graph-based |
| **MPIN (đề xuất)** | **Graph-based + Message Passing** |

---

## 5. Kết Quả Kỳ Vọng

Sau khi hoàn thành dự án, chúng ta kỳ vọng:

1. **MPIN đạt MAE và MRE thấp nhất** trên cả hai dataset, đặc biệt trong điều kiện sparsity cao (C4).
2. **MPIN-DM** (phiên bản continuous) duy trì hiệu suất tốt khi dữ liệu mới đến theo thời gian thực.
3. **Task 2**: Dữ liệu được điền bởi MPIN cho kết quả phân loại ICU tốt hơn so với các phương pháp baseline.
4. Hiểu rõ cách mô hình xử lý được 4 thách thức C1-C4 mà các baseline không giải quyết được.

---

## 6. Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Deep Learning Framework | PyTorch |
| Graph Neural Networks | PyTorch Geometric |
| Data Processing | NumPy, Pandas, SciPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Configuration | YAML |
| Experiment Tracking | Console logging |
| Notebooks | Jupyter |

---

*Tài liệu này là phần giới thiệu tổng quan. Xem thêm các tài liệu khác trong thư mục `doc/` để biết chi tiết kỹ thuật.*
