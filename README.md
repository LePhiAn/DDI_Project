DDI_PROJECT/
├── .vscode/                # Cấu hình môi trường làm việc (launch.json)
├── data/                   # Quản lý dữ liệu đầu vào và đầu ra
│   ├── mapping/            # Các file ánh xạ ID, SMILES và tác dụng phụ
│   │   ├── drug_mapping.csv
│   │   ├── full_cid_to_smiles.csv
│   │   └── side_effects_mapping.csv
│   ├── processed/          # Dữ liệu đã xử lý, sẵn sàng để train
│   │   └── ready_to_train.csv
│   └── raw/                # Dữ liệu thô (ChChSe-Decagon)
│       └── ChChSe-Decagon_polypharmacy.csv
├── models/                 # Lưu trữ trọng số mô hình R-GCN (.pth)
│   └── r_gcn_full_model.pth
├── scripts/                # Các kịch bản chạy thử nghiệm độc lập
│   └── test_predict.py
├── src/                    # Mã nguồn lõi (Core Logic & AI)
│   ├── __init__.py         # Khởi tạo package src
│   ├── auto_translate.py   # Tự động dịch thông tin liên quan
│   ├── data_loader.py      # Nạp dữ liệu đồ thị (PyG)
│   ├── data_prep.py        # Tiền xử lý dữ liệu
│   ├── fetch_names.py      # Thu thập tên thuốc từ mapping
│   ├── get_name.py         # Truy xuất tên thuốc nhanh
│   ├── inference_engine.py # Engine thực hiện suy luận logic DDI
│   ├── inference.py        # Script thực hiện dự đoán chính
│   └── model_arch.py       # Kiến trúc mạng Graph Convolutional (R-GCN)
├── web_app/                # Ứng dụng giao diện người dùng
│   ├── components/         # Các thành phần giao diện (UI)
│   │   ├── analytics_view.py   # Hiển thị phân tích và biểu đồ
│   │   ├── input_section.py    # Khu vực chọn thuốc đầu vào
│   │   ├── pair_view.py        # Xem tương tác theo cặp
│   │   ├── side_catalog_view.py # Danh mục thuốc bổ trợ
│   │   ├── sidebar.py          # Thanh điều hướng
│   │   └── single_drug_view.py  # Chi tiết từng loại thuốc
│   ├── services/           # Logic nghiệp vụ cho Web
│   │   ├── mapping_service.py  # Ánh xạ dữ liệu cho UI
│   │   ├── model_loader.py     # Tải mô hình lên ứng dụng
│   │   └── predictor_service.py # Dịch vụ gọi dự đoán
│   └── app.py              # File chạy chính của web app
├── venv/                   # Môi trường ảo Python (Virtual Environment)
├── main_process_data.py    # Luồng xử lý dữ liệu tổng thể
├── predict.py              # Script dự đoán độc lập
├── requirements.txt        # Danh sách thư viện (torch, torch-geometric, etc.)
├── README.md               # Tài liệu hướng dẫn sử dụng
└── test_notebook.ipynb     # Notebook thử nghiệm nhanh



# 1. Tạo môi trường ảo (tên là venv)
python -m venv venv

# 2. Kích hoạt môi trường
# Trên Windows:
.\venv\Scripts\activate
# Trên Mac/Linux:
source venv/bin/activate

# 3. Cài đặt các thư viện (Dùng file requirements.txt đã tạo ở bước trước)
pip install torch torch-geometric rdkit deepchem pandas flask streamlit

# 4. Chạy app
streamlit run web_app/app.py


## 🏷️ Quy tắc đặt tên trong dự án
Để dễ duy trì và tái sử dụng code, toàn bộ project giờ đã dùng một số biến chuẩn hóa chung:

- **`drug_to_id`** – ánh xạ SMILES/CID → chỉ số nguyên (trong app, inference và các script prediction).
- **`side_to_id`** – ánh xạ tên tác dụng phụ → chỉ số quan hệ.
- **`h_all`** – tensor chứa embedding của tất cả nút/do thám được tính trước.
- Các DataFrame chính thường được gọi `df` hoặc `df_full` khi cần đặt tên toàn cục.

Các tên cũ như `drug_map`, `cid_to_idx`, `side_map`, `node_embeddings` đã được đổi thành các biến kể trên để đồng nhất.

 check phiên bản 
 Ctrl + Shift + P
→ Python: Select Interpreter
