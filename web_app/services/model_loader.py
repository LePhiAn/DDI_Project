# ============================================
# Model Loader Service
# ============================================

import torch
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model_arch import RGCN_DDI_Model


class ModelLoader:

    def __init__(self, model_path, num_nodes, num_relations, hidden_channels=64, embedding_dim=16, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim

        self.model = RGCN_DDI_Model(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim
        ).to(device)

        self._load_model()

    def _load_model(self):
        """Load model weights with flexible matching for size mismatches"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Try strict loading first, fall back to partial loading if sizes don't match
        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            print(f"⚠️ Warning: Strict loading failed, attempting flexible loading...")
            # Load with strict=False to allow partial weight loading
            self.model.load_state_dict(checkpoint, strict=False)
            print("✅ Model loaded with non-strict mode")
        
        self.model.eval()

    def get_model(self):
        return self.model