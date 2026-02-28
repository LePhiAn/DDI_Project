import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
from torch_geometric.nn import RGCNConv
from tqdm import tqdm # Đảm bảo đã pip install tqdm

# 1. THIẾT LẬP THIẾT BỊ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
print(f"🚀 GPU: {torch.cuda.get_device_name(0)} - Đã sẵn sàng!")

# 2. MÔ HÌNH (Tối ưu cho VRAM 4GB)
class RGCNPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super(RGCNPredictor, self).__init__()
        self.conv1 = RGCNConv(16, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

    def forward_emb(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type).relu()
        h = self.conv2(h, edge_index, edge_type)
        return h

# 3. LOAD DATA
df = pd.read_csv('data/processed/ready_to_train.csv')
drug_map = {id: i for i, id in enumerate(pd.concat([df['SMILES_1'], df['SMILES_2']]).unique())}
side_map = {name: i for i, name in enumerate(df['Side_Name'].unique())}
num_nodes, num_relations = len(drug_map), len(side_map)

pos_edge_index = torch.tensor([[drug_map[d1] for d1 in df['SMILES_1']], 
                               [drug_map[d2] for d2 in df['SMILES_2']]], dtype=torch.long).to(cpu)
pos_edge_type = torch.tensor([side_map[s] for s in df['Side_Name']], dtype=torch.long).to(cpu)
x = torch.randn((num_nodes, 16)).to(device)

model = RGCNPredictor(num_nodes, num_relations, 16).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

# 4. VÒNG LẶP HUẤN LUYỆN VỚI THANH TIẾN ĐỘ THÔNG MINH
epochs = 500
batch_size = 1024
start_time = time.time()

print(f"🕒 Bắt đầu train 500 Epochs. Theo dõi ETA bên dưới:")

# Cấu hình thanh tiến trình (Progress Bar)
pbar = tqdm(range(1, epochs + 1), 
            desc="🚀 Tiến độ", 
            unit="ep", 
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}') # Định dạng thanh hiển thị đẹp hơn

for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    
    # Tính Embedding (Hybrid)
    e_idx_gpu, e_type_gpu = pos_edge_index.to(device), pos_edge_type.to(device)
    h_all = model.forward_emb(x, e_idx_gpu, e_type_gpu)
    del e_idx_gpu, e_type_gpu
    torch.cuda.empty_cache()
    
    epoch_loss = 0
    perm = torch.randperm(pos_edge_index.size(1))
    
    for i in range(0, pos_edge_index.size(1), batch_size):
        idx = perm[i:i + batch_size]
        b_src, b_dst_pos = pos_edge_index[0, idx].to(device), pos_edge_index[1, idx].to(device)
        b_dst_neg = torch.randint(0, num_nodes, (len(idx),), device=device)
        b_types = pos_edge_type[idx].to(device)
        
        r_emb = model.rel_emb(b_types)
        pos_out = torch.sum(h_all[b_src] * r_emb * h_all[b_dst_pos], dim=1)
        neg_out = torch.sum(h_all[b_src] * r_emb * h_all[b_dst_neg], dim=1)
        
        loss = criterion(torch.cat([pos_out, neg_out]), 
                         torch.cat([torch.ones(pos_out.size(0), device=device)*0.9, 
                                    torch.zeros(neg_out.size(0), device=device)]))
        loss.backward(retain_graph=True)
        epoch_loss += loss.item()

    optimizer.step()
    del h_all
    torch.cuda.empty_cache()
    
    # CẬP NHẬT ETA VÀ THÔNG SỐ
    vram_used = torch.cuda.memory_reserved() / 1024**2
    pbar.set_postfix({
        "Loss": f"{epoch_loss:.2f}", 
        "VRAM": f"{vram_used:.0f}MB",
        "Runtime": f"{(time.time()-start_time)/60:.1f}m"
    })

torch.save(model.state_dict(), 'models/r_gcn_full_model.pth')
print(f"\n✅ Đã xong! Tổng thời gian: {(time.time()-start_time)/60:.1f} phút.")