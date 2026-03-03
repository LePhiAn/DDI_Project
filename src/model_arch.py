import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import os

class RGCN_DDI_Model(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels=64, embedding_dim=16, node_features_dim=None):
        super(RGCN_DDI_Model, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.node_features_dim = node_features_dim
        
        # CHẾ ĐỘ V1: Dùng ID thuốc (Embedding)
        if node_features_dim is None:
            self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
            in_dim = embedding_dim
        # CHẾ ĐỘ V2: Dùng đặc trưng hóa học (SMILES/Fingerprint)
        else:
            self.preprocess = nn.Linear(node_features_dim, embedding_dim)
            in_dim = embedding_dim

        self.conv1 = RGCNConv(in_dim, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_relations)
        )

    def forward(self, edge_index, edge_type, drug1_idx=None, drug2_idx=None, x_features=None):
        # Lấy đặc trưng đầu vào
        if self.node_features_dim is None:
            x = self.node_embedding.weight
        else:
            x = F.relu(self.preprocess(x_features))
        
        # Lan truyền đồ thị
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        
        # Nếu có chỉ định cặp thuốc, thực hiện dự đoán
        if drug1_idx is not None and drug2_idx is not None:
            h1 = x[drug1_idx]
            h2 = x[drug2_idx]
            combined = torch.cat([h1, h2], dim=-1)
            return self.classifier(combined)
        
        # Nếu không, trả về toàn bộ embeddings (dùng cho inference)
        return x

# HÀM BỔ TRỢ: Tự động lưu và tải mô hình để học tiếp
def save_checkpoint(model, optimizer, epoch, loss, path='models/r_gcn_full_model.pth'):
    os.makedirs('models', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'node_features_dim': model.node_features_dim
    }, path)
    print(f"💾 Đã lưu trạng thái học tập tại epoch {epoch}")

def load_checkpoint(model, optimizer, path='models/r_gcn_full_model.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"🎬 Đã nạp thành công bộ não cũ (Epoch {checkpoint['epoch']}). Sẵn sàng học tiếp!")
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')