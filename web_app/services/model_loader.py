# ============================================
# Model Loader Service — dùng đúng kiến trúc RGCNPredictor
# (khớp với codeapp.py và checkpoint đã train)
# ============================================

import torch
import torch.nn as nn
import os


class RGCNPredictor(nn.Module):
    """Kiến trúc RGCN giống hệt codeapp.py — keys: node_emb, conv1, conv2, rel_emb"""
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super(RGCNPredictor, self).__init__()
        from torch_geometric.nn import RGCNConv
        self.node_emb = nn.Embedding(num_nodes, 16)
        self.conv1 = RGCNConv(16, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        h = self.conv1(x, edge_index, edge_type).relu()
        h = self.conv2(h, edge_index, edge_type)
        return h


class ModelLoader:

    def __init__(self, model_path, num_nodes, num_relations, hidden_channels=64,
                 embedding_dim=16, device="cpu"):
        self.model_path = os.path.abspath(model_path)
        self.device = device

        self.model = RGCNPredictor(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels
        ).to(device)

        self._load_model()

    def _load_model(self):
        """Load state_dict trực tiếp từ checkpoint (thuần state_dict, không phải dict lồng)"""
        if not os.path.exists(self.model_path):
            print(f"⚠️ Model không tìm thấy: {self.model_path} — dùng trọng số ngẫu nhiên.")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Checkpoint có thể là state_dict thuần hoặc dict có 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)
            print("✅ Model loaded thành công (strict mode)")
        except Exception as e:
            print(f"⚠️ Strict load thất bại: {e}\n⟶ Thử non-strict...")
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Model loaded (non-strict mode)")
            except Exception as e2:
                print(f"❌ Load model thất bại hoàn toàn: {e2}")

        self.model.eval()

    def get_model(self):
        return self.model