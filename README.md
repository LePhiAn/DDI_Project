# 💊 AI Pharma - Drug-Drug Interaction Interaction (DDI) Insights

## 🌟 Tổng quan dự án
Dự án **AI Pharma** là một hệ thống phân tích và dự đoán tương tác thuốc (Drug-Drug Interaction - DDI) dựa trên mô hình học máy đồ thị thông tin (**Relational Graph Convolutional Network - R-GCN**). Hệ thống cho phép người dùng kiểm tra các tương tác thuốc đã biết trong y văn và dự đoán các rủi ro tiềm ẩn giữa các cặp thuốc chưa được ghi nhận chính thức.

---

## 🏗️ Kiến trúc hệ thống
Hệ thống được thiết kế theo cấu trúc modular, chia làm 3 phần chính:
1.  **Core Engine (`src/`)**: Chứa logic huấn luyện, kiến trúc mô hình AI, và engine suy luận.
2.  **Web Application (`web_app/`)**: Giao diện người dùng xây dựng bằng Streamlit, cung cấp các dashboard trực quan.
3.  **Data & Models**: Quản lý dữ liệu thô, dữ liệu đã xử lý (SMILES, mapping) và trọng số mô hình đã huấn luyện.

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Chuẩn bị môi trường
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Cài đặt thư viện cần thiết
pip install torch torch-geometric rdkit deepchem pandas pubchempy deep-translator matplotlib streamlit
```

### 2. Chạy ứng dụng Web
```bash
streamlit run web_app/app.py
```

---

## 📂 Chi tiết cấu trúc thư mục và File `.py`

### 1. Thư mục Gốc (Root)
*   **`main_process_data.py`**:
    *   **Chức năng**: Luồng xử lý dữ liệu tổng thể từ dữ liệu thô (Decagon) sang dữ liệu sẵn sàng huấn luyện.
    *   **Biến chính**: `RAW_DATA_PATH`, `MAPPING_PATH`, `OUTPUT_PATH`.
*   **`predict.py`**:
    *   **Chức năng**: Script độc lập để chạy dự đoán DDI nhanh qua dòng lệnh (CLI).
    *   **Hàm chính**: `predict_interaction(smiles_1, smiles_2, side_effect_name)` - Trả về xác suất tương tác.
*   **`codeapp.py`**:
    *   **Chức năng**: Phiên bản "Monolithic" của ứng dụng Streamlit (tất cả trong một file). Dùng để thử nghiệm nhanh hoặc demo đơn giản.

### 2. Thư mục Lõi (`src/`)
Chứa toàn bộ logic AI và xử lý dữ liệu nền tảng.

*   **`model_arch.py`**:
    *   **Lớp chính**: `RGCN_DDI_Model` - Kiến trúc mạng nơ-ron đồ thị.
    *   **Hàm**: `forward(edge_index, edge_type, ...)` - Lan truyền thông tin qua các nút và cạnh đồ thị.
*   **`inference_engine.py`**:
    *   **Lớp chính**: `InferenceEngine` - Bộ não thực hiện suy luận rủi ro.
    *   **Hàm**: `predict_pair(drug1, drug2)` - Phân tích toàn diện rủi ro giữa 2 thuốc.
*   **`risk_assessor.py`**:
    *   **Lớp chính**: `RiskAssessor` - Đánh giá mức độ rủi ro y khoa (LOW, MODERATE, HIGH, CRITICAL).
    *   **Hàm**: `classify_risk(probability)`, `generate_alert(probability, side_effect)`.
*   **`explainer.py`**:
    *   **Lớp chính**: `DDIExplainer` - Giải thích kết quả AI dưới dạng ngôn ngữ tự nhiên.
    *   **Hàm**: `aggregate_pair_risk(predictions)`, `generate_explanation(top_predictions)`.
*   **`data_loader.py`**:
    *   **Hàm**: `create_pyg_graph(df)` - Chuyển dữ liệu CSV thành định dạng đồ thị PyTorch Geometric.
*   **`data_prep.py`**:
    *   **Hàm**: `fetch_smiles_from_pubchem(cid_list)` - Tự động tải cấu trúc SMILES từ PubChem qua CID.
*   **`auto_translate.py`**:
    *   **Chức năng**: Tự động dịch tên tác dụng phụ từ tiếng Anh sang tiếng Việt bằng Google Translate API.

### 3. Thư mục Web App (`web_app/`)
Chứa giao diện người dùng và các dịch vụ hỗ trợ UI.

#### **Services (`web_app/services/`)**
*   **`mapping_service.py`**:
    *   **Chức năng**: Quản lý ánh xạ giữa ID thuốc, SMILES, tên thuốc tiếng Anh và tiếng Việt.
    *   **Biến**: `drug_to_id`, `side_to_id`, `drug_to_name`, `side_to_vn`.
*   **`predictor_service.py`**:
    *   **Chức năng**: Cung cấp các hàm dự đoán tối ưu hóa cho giao diện (Pre-compute embeddings).
    *   **Hàm quan trọng**:
        *   `get_all_side_probs()`: Tính xác suất cho tất cả 1,300+ tác dụng phụ cùng lúc.
        *   `get_top_unknown_pairs_for_side()`: Tìm cặp thuốc rủi ro nhất cho một triệu chứng cụ thể.
*   **`model_loader.py`**:
    *   **Chức năng**: Tải mô hình R-GCN từ checkpoint `.pth` một cách an toàn.

#### **Components (`web_app/components/`)**
*   **`sidebar.py`**: Quản lý bảng điều khiển bên trái (Random cặp, bộ lọc xác suất, thống kê).
*   **`input_section.py`**: Khu vực chọn thuốc và tác dụng phụ chính ở giữa màn hình.
*   **`pair_view.py`**: Hiển thị so sánh chi tiết giữa 2 thuốc (Hình vẽ phân tử, Gauge rủi ro, Biểu đồ rủi ro tiềm ẩn).
*   **`single_drug_view.py`**: Hiển thị thông tin và biểu đồ tròn rủi ro của từng loại thuốc đơn lẻ.
*   **`analytics_view.py`**: Cung cấp các biểu đồ về hiệu năng mô hình (Loss, Accuracy nếu có).
*   **`side_catalog_view.py`**: Hiển thị danh mục tất cả tác dụng phụ có trong hệ thống.

---

## 🛠️ Chi tiết Hàm & Biến quan trọng

### 🔹 PredictorService (`web_app/services/predictor_service.py`)
| Hàm / Biến | Mô tả |
| :--- | :--- |
| `h_all` | Biến chứa toàn bộ Embedding của thuốc sau khi đi qua mô hình AI. |
| `_distmult_score(...)` | Tính toán điểm tương tác giữa 2 nút thuốc dựa trên một quan hệ (side effect). |
| `_score_to_prob(score)` | Dùng hàm *tanh* và *sigmoid scaling* để chuyển điểm thô thành xác suất (0-100%). |
| `get_prob(d1, d2, side)` | Trả về xác suất của một cặp thuốc với một triệu chứng cụ thể. |

### 🔹 InferenceEngine (`src/inference_engine.py`)
| Hàm | Đầu vào | Đầu ra |
| :--- | :--- | :--- |
| `predict_pair` | `drug1_smiles`, `drug2_smiles` | JSON chứa mức độ rủi ro, phân tích lâm sàng và Top 10 rủi ro. |
| `_determine_overall_risk` | Dữ liệu tổng hợp | Phân loại rủi ro tổng hợp (CRITICAL nếu có triệu chứng cực kỳ nguy hiểm). |

### 🔹 RGCN_DDI_Model (`src/model_arch.py`)
| Lớp / Phương thức | Chi tiết |
| :--- | :--- |
| `RGCNConv` | Lớp tích chập đồ thị xử lý các mối quan hệ đa tầng. |
| `rel_emb` | Ma trận Embedding cho 1,317 loại tác dụng phụ khác nhau. |
| `classifier` | Mạng nơ-ron Dense dùng để phân loại rủi ro từ vector đặc trưng. |

---

## 🏷️ Quy tắc dữ liệu (Data Rules)
Dự án sử dụng các định danh chuẩn để đồng bộ giữa các module:
- **`drug_to_id`**: Mapping SMILES $\rightarrow$ ID số nguyên (0 đến N-1).
- **`side_to_id`**: Mapping Side Name $\rightarrow$ ID quan hệ (0 đến 1316).
- **`h_all`**: Tensor lưu trữ trạng thái "hiểu biết" của mô hình về toàn bộ dược phẩm.

---

## 📧 Liên hệ & Bản quyền
Dự án được phát triển cho mục đích nghiên cứu phân tích dữ liệu DDI. Mọi thông tin dự đoán từ AI nên được tham khảo ý kiến chuyên gia y tế trước khi áp dụng lâm sàng.
