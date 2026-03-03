import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch_geometric.nn import RGCNConv

# --- 1. ĐỊNH NGHĨA LẠI KIẾN TRÚC MODEL (Y hệt lúc train) ---
class RGCNPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super(RGCNPredictor, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, 16) 
        self.conv1 = RGCNConv(16, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        h = self.conv1(x, edge_index, edge_type).relu()
        h = self.conv2(h, edge_index, edge_type)
        return h

# --- 2. ĐƯỜNG DẪN FILE (Đã cập nhật theo ý bạn) ---
input_path = 'data/processed/ready_to_train.csv' #
model_path = 'models/r_gcn_full_model.pth'      #
mapping_path = 'data/mapping/drug_mapping.csv'

# Kiểm tra file đầu vào
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {input_path}")

# --- 3. CHUẨN BỊ DỮ LIỆU & MAPPING ---
df = pd.read_csv(input_path) 
all_smiles = pd.concat([df['SMILES_1'], df['SMILES_2']]).unique()
# canonical mapping variable
drug_to_id = {smiles: i for i, smiles in enumerate(all_smiles)}

# Load tên thuốc từ mapping (nếu có) để in ra cho đẹp
drug_name_dict = {}
if os.path.exists(mapping_path):
    mapping_df = pd.read_csv(mapping_path)
    drug_name_dict = dict(zip(mapping_df['SMILES'], mapping_df['Drug_Name']))

side_to_id = {name: i for i, name in enumerate(df['Side_Name'].unique())}
num_nodes = len(drug_to_id)
num_relations = len(side_to_id)

# Load edge_index gốc để Model tính Embeddings
pos_edge_index = torch.tensor([
    [drug_to_id[d1] for d1 in df['SMILES_1']],
    [drug_to_id[d2] for d2 in df['SMILES_2']]
], dtype=torch.long)
pos_edge_type = torch.tensor([side_to_id[s] for s in df['Side_Name']], dtype=torch.long)

# --- 4. KHỞI TẠO & NẠP WEIGHTS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCNPredictor(num_nodes, num_relations, 64).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Đã nạp thành công model từ {model_path}")
else:
    print(f"❌ Cảnh báo: Không tìm thấy file model tại {model_path}")

# --- 5. HÀM DỰ ĐOÁN (PREDICT) ---
def predict_interaction(smiles_1, smiles_2, side_effect_name):
    if smiles_1 not in drug_to_id or smiles_2 not in drug_to_id:
        return "❌ Lỗi: Thuốc không nằm trong dữ liệu huấn luyện."
    
    if side_effect_name not in side_to_id:
        return f"❌ Lỗi: Tác dụng phụ '{side_effect_name}' không tồn tại."

    idx1, idx2 = drug_to_id[smiles_1], drug_to_id[smiles_2]
    rel_idx = side_to_id[side_effect_name]

    with torch.no_grad():
        h = model(pos_edge_index.to(device), pos_edge_type.to(device))
        r_emb = model.rel_emb(torch.tensor([rel_idx]).to(device))
        score = torch.sum(h[idx1] * r_emb * h[idx2], dim=1)
        probability = torch.sigmoid(score).item() 
        
    return probability

# --- 6. CHẠY THỬ NGHIỆM ---
test_s1 = df['SMILES_1'].iloc[0] 
test_s2 = df['SMILES_2'].iloc[0]
test_side = df['Side_Name'].iloc[0]

# Lấy tên thuốc từ từ điển (nếu không có thì dùng SMILES)
name1 = drug_name_dict.get(test_s1, test_s1)
name2 = drug_name_dict.get(test_s2, test_s2)

prob = predict_interaction(test_s1, test_s2, test_side)

print(f"\n" + "="*40)
print(f"💊 DỰ ĐOÁN TƯƠNG TÁC THUỐC")
print(f"-"*40)
print(f"Thuốc 1: {name1}")
print(f"Thuốc 2: {name2}")
print(f"Tác dụng phụ: {test_side}")
print(f"Xác suất: {prob*100:.2f}%")
print(f"="*40)