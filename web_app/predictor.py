import sys
import os
import torch
import pandas as pd

# Thêm thư mục gốc (DDI_Project) vào PYTHONPATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.model_arch import RGCN_DDI_Model


class DDIPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ====== ĐƯỜNG DẪN TUYỆT ĐỐI AN TOÀN ======
        self.model_path = os.path.join(BASE_DIR, 'models', 'r_gcn_checkpoint.pth')
        self.mapping_path = os.path.join(BASE_DIR, 'data', 'mapping', 'full_cid_to_smiles.csv')
        self.processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'ready_to_train.csv')

        # ====== 1. LOAD MAPPING ======
        self.mapping_df = pd.read_csv(self.mapping_path)

        # Số node = số CID
        num_nodes = len(self.mapping_df)

        # Số relation của Decagon
        num_relations = 464

        # ====== 2. KHỞI TẠO MODEL ======
        self.model = RGCN_DDI_Model(
            num_nodes=num_nodes,
            num_relations=num_relations
        ).to(self.device)

        # ====== 3. LOAD CHECKPOINT ======
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        print("✅ Model loaded successfully!")

    # ===============================
    # LẤY DANH SÁCH THUỐC
    # ===============================
    def get_drug_list(self):
        return self.mapping_df['CID'].tolist()

    # ===============================
    # HÀM DỰ ĐOÁN (TẠM THỜI DUMMY)
    # ===============================
    def predict(self, drug_a, drug_b, side_effect_id):
        # TODO: Viết logic inference đúng theo graph data
        # Hiện tại trả dummy để test web
        return 0.88