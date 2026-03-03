# src/inference.py

import torch
import pandas as pd
import os
from src.model_arch import RGCN_DDI_Model  # dùng model bạn đã cung cấp

class RGCNInference:
    def __init__(
        self,
        data_path="data/processed/ready_to_train.csv",
        model_path="models/r_gcn_full_model.pth",
        device=None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Load Data ---
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        self.df = pd.read_csv(data_path)

        # --- Build Mapping (using canonical names drug_to_id / side_to_id) ---
        all_smiles = pd.concat(
            [self.df["SMILES_1"], self.df["SMILES_2"]]
        ).unique()

        self.drug_to_id = {smiles: i for i, smiles in enumerate(all_smiles)}
        self.side_to_id = {
            name: i for i, name in enumerate(self.df["Side_Name"].unique())
        }

        self.num_nodes = len(self.drug_to_id)
        self.num_relations = len(self.side_to_id)

        # --- Build Graph ---
        # build graph tensors using canonical mapping names
        self.edge_index = torch.tensor(
            [
                [self.drug_to_id[d1] for d1 in self.df["SMILES_1"]],
                [self.drug_to_id[d2] for d2 in self.df["SMILES_2"]],
            ],
            dtype=torch.long,
        ).to(self.device)

        self.edge_type = torch.tensor(
            [self.side_to_id[s] for s in self.df["Side_Name"]],
            dtype=torch.long,
        ).to(self.device)

        # --- Load Model ---
        self.model = RGCN_DDI_Model(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        # --- Precompute Embeddings (quan trọng) ---
        with torch.no_grad():
            # use the same name h_all that other modules expect
            self.h_all = self.model(
                self.edge_index, self.edge_type
            )

    # --------------------------------------------------

    def predict(self, smiles_1, smiles_2, side_effect_name):
        # check using the canonical maps
        if smiles_1 not in self.drug_to_id:
            raise ValueError("Drug 1 not in training data.")

        if smiles_2 not in self.drug_to_id:
            raise ValueError("Drug 2 not in training data.")

        if side_effect_name not in self.side_to_id:
            raise ValueError("Side effect not found.")

        idx1 = self.drug_to_id[smiles_1]
        idx2 = self.drug_to_id[smiles_2]
        rel_idx = self.side_to_id[side_effect_name]

        with torch.no_grad():
            h1 = self.h_all[idx1]
            h2 = self.h_all[idx2]
            r = self.model.rel_emb(
                torch.tensor([rel_idx]).to(self.device)
            )

            score = torch.sum(h1 * r.squeeze(0) * h2)
            prob = torch.sigmoid(score).item()

        return prob