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

# Cài đặt toàn bộ thư viện từ file requirements
pip install -r requirements.txt
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
*   **`fetch_names.py`**:
    *   **Chức năng**: Tự động tra cứu tên thuốc thương mại (Synonyms) từ PubChem dựa trên SMILES.
*   **`get_name.py`**: 
    *   **Chức năng**: Script tiện ích để tra cứu nhanh 1 tên thuốc đơn lẻ.
*   **`inference.py`**:
    *   **Chức năng**: Cung cấp các hàm nền tảng để tính toán xác suất tương tác từ Vector Embedding.
*   **`auto_translate.py`**:
    *   **Chức năng**: Tự động dịch tên tác dụng phụ từ tiếng Anh sang tiếng Việt bằng Google Translate API.

### 3. Thư mục Web App (`web_app/`)
Chứa giao diện người dùng và các dịch vụ hỗ trợ UI.

#### **Services (`web_app/services/`)**
*   **`mapping_service.py`**:
    *   **Chức năng**: Quản lý ánh xạ giữa ID thuốc, SMILES, tên thuốc tiếng Anh và tiếng Việt.
*   **`predictor_service.py`**:
    *   **Chức năng**: Dự đoán tối ưu cho UI (Pre-compute embeddings).
*   **`model_loader.py`**:
    *   **Chức năng**: Tải mô hình R-GCN từ file `.pth`.

#### **Components (`web_app/components/`)**
*   **`sidebar.py`**, **`input_section.py`**, **`pair_view.py`**, **`single_drug_view.py`**, **`analytics_view.py`**, **`side_catalog_view.py`**: Các thành phần giao diện chia nhỏ theo module.

---

## 🎨 Chi tiết Biến số & Kiến trúc Giao diện (Streamlit UI)
Phần này dành riêng cho việc phát triển và nâng cấp giao diện `app.py`.

### 1. Các biến dữ liệu nền tảng (Core Data)
Các biến này được load qua `load_all()` và dùng chung cho toàn bộ app:
| Tên biến | Kiểu | Chức năng |
| :--- | :--- | :--- |
| `mapping` | Service | Chứa logic ánh xạ (SMILES, Tên thuốc, ID). |
| `predictor` | Service | Chứa model AI và thực hiện dự đoán xác suất. |
| `df_full` | DataFrame | Bảng dữ liệu gốc (Ground Truth) từ dataset Decagon. |
| `drug_list` | List | Danh sách tên thuốc để hiển thị trong `st.selectbox`. |
| `side_list` | List | Danh sách 1,300+ tác dụng phụ. |
| `drug_names` | Dict | Map SMILES $\rightarrow$ Tên thuốc thường gọi. |
| `side_vn_map` | Dict | Map Tên tiếng Anh $\rightarrow$ Tên tiếng Việt. |

### 2. Các biến điều khiển & Trạng thái (Control & State)
Quản lý qua Sidebar và Session State:
| Tên biến | Chức năng |
| :--- | :--- |
| `show_top_10` | Bật/tắt biểu đồ Thống kê Top 10 tác dụng phụ. |
| `show_extreme`| Bật/tắt bảng "Top 20 cặp thuốc nguy hiểm nhất". |
| `prob_threshold`| Ngưỡng lọc rủi ro AI (mặc định 50%). |
| `st.session_state.d1` | Lưu thuốc thứ 1 người dùng đang chọn. |
| `st.session_state.d2` | Lưu thuốc thứ 2 người dùng đang chọn. |
| `st.session_state.side` | Lưu tác dụng phụ đang được chọn để soi. |

### 3. Logic luồng hiển thị (UI Flow Logic)
Giao diện thay đổi dựa trên trạng thái của `d1` và `d2`:
1.  **Mặc định (`Trống` & `Trống`)**: Hiển thị **Side Catalog** (Danh mục tổng).
2.  **Soi 1 thuốc (`d1` hoặc `d2` khác `Trống`)**: Gọi `render_single_drug_view`.
3.  **Soi cặp thuốc (Cả 2 đều khác `Trống`)**: Gọi `render_pair_view`.
4.  **Lọc theo Triệu chứng**: Nếu chọn 1 `selected_side` cụ thể, hệ thống sẽ ưu tiên lọc các cặp thuốc liên quan đến triệu chứng đó.

---

## ☁️ Hướng dẫn Deploy lên Streamlit Cloud
Để ứng dụng chạy online, hãy thực hiện các bước sau:

1.  **Repository**: Đẩy toàn bộ source code lên GitHub (bao gồm cả thư mục `models/` và `data/`).
2.  **Entry Point**: Khi cấu hình trên Streamlit Cloud, trỏ đường dẫn chính vào: `web_app/app.py`.
3.  **Requirements**: Streamlit Cloud sẽ tự động đọc file `requirements.txt` ở thư mục gốc để cài đặt môi trường.
4.  **Secrets (Nếu có)**: Nếu bạn dùng các API key bí mật, hãy cấu hình trong mục "Secrets" của Streamlit.

---

## 🛠️ Chi tiết Hàm & Biến quan trọng

### 🔹 PredictorService (`web_app/services/predictor_service.py`)
| Hàm / Biến | Mô tả |
| :--- | :--- |
| `h_all` | Tensor chứa toàn bộ Embedding của thuốc sau khi đi qua mô hình AI. |
| `get_prob(d1, d2, side)` | Trả về xác suất của một cặp thuốc với một triệu chứng cụ thể. |

### 🔹 InferenceEngine (`src/inference_engine.py`)
| Hàm | Đầu vào | Đầu ra |
| :--- | :--- | :--- |
| `predict_pair` | `drug1_smiles`, `drug2_smiles` | JSON chứa mức độ rủi ro và phân tích lâm sàng. |

---

## 🏷️ Quy tắc dữ liệu (Data Rules)
Dự án sử dụng các định danh chuẩn để đồng bộ giữa các module:
- **`data/raw/`**: Chứa file `ChChSe-Decagon_polypharmacy.csv` (Dữ liệu gốc).
- **`data/mapping/`**: Chứa `drug_mapping.csv` (Tên thuốc) và `full_cid_to_smiles.csv`.
- **`data/processed/`**: Chứa `ready_to_train.csv` (Dữ liệu đã sạch).
- **`models/`**: Chứa file `r_gcn_full_model.pth` (Trọng số mô hình).

---

## 📧 Liên hệ & Bản quyền
Dự án được phát triển cho mục đích nghiên cứu phân tích dữ liệu DDI. Mọi thông tin dự đoán từ AI nên được tham khảo ý kiến chuyên gia y tế trước khi áp dụng lâm sàng.

