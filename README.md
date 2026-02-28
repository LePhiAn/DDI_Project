DDI_Project/
├── data/ 
│   ├── raw/                # File ChChSe-Decagon_miner-twosides.tsv
│   │   └── ChChSe-Decagon_polypharmacy.csv  
│   ├── processed/          # File sau khi map SMILES
│   │   └── ready_to_train.csv  
│   └── mapping/
│   │   └── full_cid_to_smiles.csv 
│   
├── models/                 # File .pth
│   └──  r_gcn_full_model.pth
├── src/                    # TRÁI TIM CỦA DỰ ÁN
│   ├── __init__.py
│   ├── data_prep.py        # Gộp mapping.py và data_loader.py vào đây cho gọn
│   ├── data_loader.py
│   └── model_arch.py       # Khai báo R-GCN
├── web_app/                # GIAO DIỆN
│   ├── app.py              # Gọi src.model_arch để dự đoán
│   └── static/             # CSS, hình ảnh
├── .gitignore              # Để không upload data nặng lên Github
├── requirements.txt
├── main_train_custom.py    # File script để train (thay cho notebook nếu muốn chạy nhanh)
├── main_process_data       # Kết hợp với deepchem để tạo file ready_to_train.csv
└── test_notebook.ipynb     # Để ngay ngoài gốc để tiện import src



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
