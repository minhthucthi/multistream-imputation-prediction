# 02 – Lựa Chọn Mô Hình: Tại Sao Chọn MPIN?

---

## 1. Bối Cảnh Lựa Chọn

Có nhiều phương pháp điền giá trị thiếu (missing value imputation) đã được đề xuất trong văn liệu. Mỗi phương pháp có ưu và nhược điểm riêng. Phần này phân tích **lý do chọn MPIN** và tại sao các phương pháp khác không đủ phù hợp cho bài toán của chúng ta.

---

## 2. Tóm Tắt 4 Thách Thức Cần Giải Quyết

Nhắc lại 4 thách thức (C1–C4) đặc thù của dữ liệu luồng cảm biến đa thuộc tính:

| Thách thức | Mô tả chi tiết |
|-----------|---------------|
| **C1 – Aperiodicity** | Dữ liệu không được đo theo chu kỳ đều đặn. Mỗi thuộc tính có tần suất đo riêng, timestamps không đồng bộ. Phương pháp giả định dữ liệu tuần hoàn sẽ thất bại. |
| **C2 – Concurrency** | Nhiều luồng (J) hoạt động song song cùng lúc. Thông tin giữa các luồng có thể tương quan và bổ sung lẫn nhau. Phương pháp xử lý độc lập từng luồng bỏ sót thông tin này. |
| **C3 – Heterogeneity** | Các thuộc tính có đơn vị đo khác nhau (°C, mmHg, mg/L...), phân phối khác nhau, và ý nghĩa ngữ nghĩa khác nhau. Phương pháp không phân biệt sẽ bị ảnh hưởng bởi scale. |
| **C4 – Sparsity** | Tỷ lệ thiếu rất cao (ICU: 80%, Wi-Fi: 85.6%). Phương pháp dựa nhiều vào dữ liệu quan sát có sẵn sẽ thiếu thông tin để điền chính xác. |

---

## 3. So Sánh Chi Tiết Các Phương Pháp

### 3.1 Bảng So Sánh Tổng Quan

| Phương pháp | Loại | C1 | C2 | C3 | C4 | Streaming | Phức tạp |
|-------------|------|----|----|----|----|-----------|----------|
| **MEAN** | Statistical | ❌ | ❌ | ⚠️ | ❌ | ✅ | Rất thấp |
| **KNN** | Instance-based | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | Thấp |
| **MICE** | Iterative | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | Trung bình |
| **MF** | Low-rank | ❌ | ⚠️ | ❌ | ⚠️ | ❌ | Trung bình |
| **BRITS** | RNN | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ | Cao |
| **SAITS** | Attention | ✅ | ❌ | ⚠️ | ⚠️ | ❌ | Cao |
| **FeaProp (FP)** | Graph | ⚠️ | ✅ | ⚠️ | ⚠️ | ❌ | Trung bình |
| **MPIN** | Graph + MsgProp | ✅ | ✅ | ✅ | ✅ | ✅ | Cao |

*Chú thích: ✅ = Giải quyết tốt, ⚠️ = Giải quyết một phần, ❌ = Không giải quyết*

---

### 3.2 Phân Tích Từng Phương Pháp

#### 🔴 MEAN Imputation

**Mô tả**: Thay thế giá trị thiếu bằng giá trị trung bình của thuộc tính đó theo toàn bộ dataset.

$$\hat{x}_{j,t,d} = \frac{1}{|\{(j',t') : M_{j',t',d}=1\}|} \sum_{(j',t') : M_{j',t',d}=1} x_{j',t',d}$$

**Ưu điểm**:
- Cực kỳ đơn giản, không cần train.
- Nhanh, phù hợp làm baseline tối thiểu.

**Nhược điểm**:
- ❌ **C1**: Không tính đến thời gian – giá trị mean toàn cục không phản ánh xu hướng theo thời gian.
- ❌ **C2**: Không tận dụng thông tin từ các luồng khác.
- ❌ **C4**: Với sparsity 80%, chỉ 20% điểm dữ liệu tạo nên mean – không đại diện.
- Làm **giảm phương sai** của dữ liệu, bóp méo phân phối.

**Kết luận**: Chỉ dùng làm lower bound baseline.

---

#### 🔴 KNN Imputation

**Mô tả**: Với mỗi điểm dữ liệu bị thiếu, tìm K láng giềng gần nhất (theo Euclidean distance trên các thuộc tính quan sát được) và lấy trung bình có trọng số.

**Ưu điểm**:
- Tính đến cấu trúc local của dữ liệu.
- Không giả định phân phối cụ thể.

**Nhược điểm**:
- ❌ **C1**: Khoảng cách Euclidean trên time series không đồng bộ không có ý nghĩa rõ ràng.
- ⚠️ **C2**: Có thể so sánh giữa các luồng, nhưng không có cơ chế lan truyền thông tin có hướng.
- ❌ **C4**: Với sparsity cao, khoảng cách tính trên rất ít thuộc tính quan sát – không đáng tin cậy.
- **Phức tạp tính toán**: O(n²D) – không scale với dataset lớn (J=11,988).

**Kết luận**: Hiệu suất tốt hơn MEAN nhưng không scale và không xử lý được sparsity cao.

---

#### 🟡 MICE (Multiple Imputation by Chained Equations)

**Mô tả**: Mô hình hóa phân phối có điều kiện của mỗi thuộc tính dựa trên các thuộc tính khác, lặp đi lặp lại cho đến hội tụ.

**Ưu điểm**:
- Tính đến tương quan giữa các thuộc tính.
- Có lý thuyết thống kê vững chắc.

**Nhược điểm**:
- ❌ **C1**: Giả định dữ liệu i.i.d, không phù hợp với time series.
- ❌ **C2**: Chỉ xử lý tương quan nội bộ trong một instance, không giữa các luồng.
- ❌ **Streaming**: Rất chậm (nhiều vòng lặp), không phù hợp real-time.
- ⚠️ **C4**: Với 80% thiếu, hội tụ không ổn định.

**Kết luận**: Phù hợp cho dữ liệu tabular thông thường, không phù hợp cho streaming sensor data.

---

#### 🟡 Matrix Factorization (MF)

**Mô tả**: Phân tích ma trận $X \approx U \cdot V^T$ tìm biểu diễn low-rank, sau đó dùng để điền giá trị thiếu.

**Ưu điểm**:
- Nắm bắt được cấu trúc latent.
- Hiệu quả với collaborative filtering.

**Nhược điểm**:
- ❌ **C1**: Không có cơ chế xử lý temporal structure.
- ❌ **C3**: Giả định linear low-rank structure – không phù hợp với dữ liệu sensor phức tạp.
- ❌ **Streaming**: Cần retrain toàn bộ khi có dữ liệu mới.
- ⚠️ **C4**: Low-rank assumption có thể không đúng khi data rất sparse.

**Kết luận**: Hiệu quả cho recommendation systems, không phù hợp cho sensor streams.

---

#### 🟡 BRITS (Bidirectional Recurrent Imputation for Time Series)

**Mô tả**: Sử dụng RNN hai chiều (forward và backward) để điền giá trị thiếu trong time series, có tích hợp decay factor để xử lý missing.

**Ưu điểm**:
- ✅ **C1**: RNN tự nhiên xử lý được thứ tự thời gian không đều nhau (với decay).
- Tận dụng context temporal (quá khứ và tương lai).

**Nhược điểm**:
- ❌ **C2**: Mỗi luồng được xử lý độc lập, không có cơ chế trao đổi thông tin giữa các luồng.
- ⚠️ **C4**: RNN cần đủ dữ liệu để học temporal patterns, với 80% missing sẽ gặp khó khăn.
- ❌ **Streaming**: BRITS cần biết toàn bộ sequence trước (bidirectional), không phù hợp real-time.

**Kết luận**: Tốt cho single stream time series, nhưng thiếu khả năng cross-stream information sharing.

---

#### 🟡 SAITS (Self-Attention-based Imputation for Time Series)

**Mô tả**: Sử dụng cơ chế self-attention (Transformer) để học tương quan giữa các timesteps và thuộc tính.

**Ưu điểm**:
- ✅ **C1**: Attention mechanism tự nhiên xử lý irregular timestamps.
- Học được long-range temporal dependencies.
- State-of-the-art trên nhiều benchmark.

**Nhược điểm**:
- ❌ **C2**: Chỉ xử lý attention trong nội bộ một luồng, không cross-stream.
- ⚠️ **C4**: Với 80% missing, attention trên rất ít observed values – hiệu quả giảm sút.
- ❌ **Streaming**: Transformer cần toàn bộ sequence, không incremental.
- **Phức tạp tính toán**: O(T²) attention – không scale với window dài.

**Kết luận**: Rất mạnh cho single-stream imputation, nhưng không tận dụng được thông tin từ nhiều luồng đồng thời.

---

#### 🟠 Feature Propagation (FeaProp / FP)

**Mô tả**: Lan truyền đặc trưng trên đồ thị – dùng thông tin từ các node láng giềng để điền giá trị thiếu thông qua nhiều bước diffusion.

$$\mathbf{X}^{(l+1)} = \hat{A} \mathbf{X}^{(l)}$$

**Ưu điểm**:
- ✅ **C2**: Rõ ràng sử dụng thông tin từ nhiều node/luồng.
- Đơn giản, hiệu quả về tính toán.

**Nhược điểm**:
- ⚠️ **C1**: Đồ thị tĩnh, không thích nghi với temporal structure.
- ❌ **C3**: Lan truyền đơn giản không phân biệt được độ quan trọng của từng thuộc tính.
- ⚠️ **C4**: Nếu hầu hết láng giềng cũng bị thiếu dữ liệu → lan truyền bị lỗi.
- Không có **learnable parameters** → không tối ưu được cho bài toán cụ thể.

**Kết luận**: Ý tưởng graph-based đúng hướng nhưng thiếu khả năng học và xử lý heterogeneity.

---

## 4. Tại Sao Chọn MPIN?

MPIN (Message Propagation Imputation Network) giải quyết **tất cả 4 thách thức**:

### 4.1 Giải quyết C1 – Aperiodicity

MPIN sử dụng **tumbling windows** để chia dữ liệu thành các chunk có kích thước cố định, không giả định dữ liệu có chu kỳ. Trong mỗi chunk, MPIN xử lý toàn bộ ma trận $J \times T$ mà không cần timestamps đều nhau.

### 4.2 Giải quyết C2 – Concurrency

MPIN xây dựng **similarity graph** giữa J luồng dữ liệu và thực hiện message propagation để chia sẻ thông tin giữa các luồng có tương quan. Đây là điểm khác biệt cốt lõi so với BRITS và SAITS.

### 4.3 Giải quyết C3 – Heterogeneity

MPIN học được **correlation weight** $c(x_i, x_k)$ giữa từng cặp node, tự động điều chỉnh mức độ ảnh hưởng dựa trên sự tương đồng thực tế. Trọng số này được học từ dữ liệu, không phải định sẵn.

### 4.4 Giải quyết C4 – Sparsity

Với **transductive learning**, MPIN học trực tiếp trên chunk hiện tại (không cần train-test split lớn), tối đa hóa việc sử dụng dữ liệu có sẵn. Ngoài ra, cấu trúc graph cho phép bù đắp thiếu hụt của một luồng bằng thông tin từ các luồng láng giềng.

---

## 5. Kiến Trúc MPIN Giải Thích Từng Bước

### Bước 1: Xây dựng Data Chunk

Từ các luồng dữ liệu đến trong cửa sổ thời gian T:

$$\text{Chunk} = \{(x_{j,t}, m_{j,t}) : j = 1,...,J, t = 1,...,T\}$$

Mỗi instance $i = (j,t)$ là một vector thuộc tính D chiều với mask nhị phân.

### Bước 2: Mean-fill khởi tạo

Để xây dựng đồ thị ban đầu, MPIN điền tạm thời bằng mean:

$$\tilde{x}_{i,d} = \begin{cases} x_{i,d} & \text{if } m_{i,d} = 1 \\ \bar{x}_d & \text{if } m_{i,d} = 0 \end{cases}$$

### Bước 3: Xây dựng Similarity Graph

Tính khoảng cách Euclidean giữa tất cả cặp instances, lấy K láng giềng gần nhất (k=10):

$$\text{dist}(i, k) = \|\tilde{x}_i - \tilde{x}_k\|_2$$

$$\mathcal{N}(i) = \{k : k \in \text{K-NN}(i)\}$$

### Bước 4: Message Propagation (MsgProp Layer)

Mỗi layer cập nhật biểu diễn của node $i$ bằng cách tổng hợp thông tin từ láng giềng (GraphSAGE-style):

$$\tilde{x}_i^{(l+1)} = \sum_{k \in \mathcal{N}(i)} c(x_i^{(l)}, x_k^{(l)}) \cdot x_k^{(l)} \cdot W^{(l)}$$

Với ràng buộc: chỉ cập nhật các vị trí thiếu (bound condition):

$$x_{i,d}^{(l+1)} = \begin{cases} x_{i,d}^{(0)} & \text{if } m_{i,d} = 1 \quad \text{(giữ nguyên observed)} \\ \tilde{x}_{i,d}^{(l+1)} & \text{if } m_{i,d} = 0 \quad \text{(cập nhật missing)} \end{cases}$$

**Correlation weight** $c(x_i, x_k)$ là learnable, phản ánh độ tương đồng giữa hai instances.

### Bước 5: Stacking 2 Layers

Với 2 lớp MsgProp, thông tin có thể lan truyền qua 2-hop neighbors, mở rộng "tầm nhìn" của mỗi node trong đồ thị.

### Bước 6: Continuous Framework (MPIN-DM)

Khi cửa sổ thời gian mới đến (chunk mới):

**Data Update**: Lọc các instances quan trọng từ chunk cũ theo **importance score** $\phi$:

$$\phi_i = \frac{\text{số thuộc tính quan sát được của } i}{D}$$

Giữ lại instances có $\phi_i \geq \eta = 0.6$ để bổ sung vào chunk mới.

**Model Update**: Tiếp tục train (warm-start) MPIN trên chunk mới kết hợp instances quan trọng từ chunk cũ, sử dụng transfer learning trên message passing module.

---

## 6. Lựa Chọn Hyperparameter

| Hyperparameter | Giá trị | Lý do |
|----------------|---------|-------|
| **K** (KNN graph) | 10 | Theo bài báo gốc; đủ láng giềng mà không quá dense |
| **Số lớp MsgProp** | 2 | Trade-off giữa receptive field và over-smoothing |
| **η** (importance threshold) | 0.6 | Giữ lại instances có ít nhất 60% thuộc tính quan sát |
| **Epochs** | 100 | Đủ hội tụ cho transductive learning |
| **Learning rate** | 0.001 | Adam optimizer, standard choice |
| **Val split** | 5% | Validation nhỏ để tránh lãng phí dữ liệu sparse |
| **Loss** | MSE | Standard cho regression/imputation |

---

## 7. Lý Do Chọn Classifier Đơn Giản cho Task 2

Cho Task 2 (ICU survival prediction), chúng ta chọn **Logistic Regression** thay vì mô hình phức tạp vì:

1. **Mục tiêu là so sánh chất lượng imputation**, không phải tối ưu predictor. Classifier đơn giản giúp isolate tác động của imputation quality.
2. **ICU dataset có J=11,988 samples** – đủ cho Logistic Regression hoạt động ổn định.
3. **Interpretability**: Logistic Regression cho phép phân tích feature importance, dễ diễn giải với bác sĩ.
4. Nếu MPIN imputation tốt hơn → Logistic Regression trên dữ liệu MPIN sẽ tốt hơn → chứng minh được giá trị của imputation.

---

*Xem thêm chi tiết kỹ thuật trong `doc/03_how_it_works.md`*
