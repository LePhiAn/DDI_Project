import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import os

class RGCN_DDI_Model(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels=64, node_features_dim=None):
        super(RGCN_DDI_Model, self).__init__()
        
        self.node_features_dim = node_features_dim
        
        # CHẾ ĐỘ V1: Dùng ID thuốc (Embedding)
        if node_features_dim is None:
            self.node_embedding = nn.Embedding(num_nodes, hidden_channels)
            in_dim = hidden_channels
        # CHẾ ĐỘ V2: Dùng đặc trưng hóa học (SMILES/Fingerprint)
        else:
            self.preprocess = nn.Linear(node_features_dim, hidden_channels)
            in_dim = hidden_channels

        self.conv1 = RGCNConv(in_dim, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_relations)
        )

    def forward(self, edge_index, edge_type, drug1_idx, drug2_idx, x_features=None):
        # Lấy đặc trưng đầu vào
        if self.node_features_dim is None:
            x = self.node_embedding.weight
        else:
            x = F.relu(self.preprocess(x_features))
        
        # Lan truyền đồ thị
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        
        # Kết hợp cặp thuốc để dự đoán
        h1 = x[drug1_idx]
        h2 = x[drug2_idx]
        combined = torch.cat([h1, h2], dim=-1)
        
        return self.classifier(combined)

# HÀM BỔ TRỢ: Tự động lưu và tải mô hình để học tiếp
def save_checkpoint(model, optimizer, epoch, loss, path='models/r_gcn_checkpoint.pth'):
    os.makedirs('models', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'node_features_dim': model.node_features_dim
    }, path)
    print(f"💾 Đã lưu trạng thái học tập tại epoch {epoch}")

def load_checkpoint(model, optimizer, path='models/r_gcn_checkpoint.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"🎬 Đã nạp thành công bộ não cũ (Epoch {checkpoint['epoch']}). Sẵn sàng học tiếp!")
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')