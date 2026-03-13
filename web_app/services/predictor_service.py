# ============================================
# Predictor Service — pre-compute h_all như codeapp.py
# Sử dụng min-max normalization + tanh scaling để tránh sigmoid saturation
# ============================================

import torch
import torch.nn.functional as F
import pandas as pd
import os
import math


class PredictorService:

    def __init__(self, model, drug_to_id, side_to_id, df_full,
                 device, side_to_vn=None):
        self.model = model
        self.drug_to_id = drug_to_id
        self.side_to_id = side_to_id
        self.id_to_side = {v: k for k, v in side_to_id.items()}
        self.df_full = df_full
        self.device = device
        self.side_to_vn = side_to_vn or {}

        # Pre-compute toàn bộ node embeddings 1 lần (giống codeapp.py)
        self.h_all = self._precompute_embeddings()
        # Pre-compute tất cả relation embeddings (normalized) một lần
        self.r_embs = self._precompute_rel_embs()

    def _precompute_embeddings(self):
        """Tạo edge_index từ df_full rồi forward model 1 lần để lấy h_all."""
        u = [self.drug_to_id[s] for s in self.df_full['SMILES_1']]
        v = [self.drug_to_id[s] for s in self.df_full['SMILES_2']]
        edge_index = torch.tensor([u, v], dtype=torch.long).to(self.device)
        edge_type = torch.tensor(
            [self.side_to_id[s] for s in self.df_full['Side_Name']],
            dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            h = self.model(edge_index, edge_type)
            # L2-normalize để mỗi embedding có norm = 1
            h = F.normalize(h, p=2, dim=1)
        return h  # [num_nodes, hidden_channels]

    def _precompute_rel_embs(self):
        """Pre-compute và normalize tất cả relation embeddings."""
        n = len(self.side_to_id)
        idx = torch.arange(n, dtype=torch.long).to(self.device)
        with torch.no_grad():
            r = self.model.rel_emb(idx)  # [n, hidden_dim]
            r = F.normalize(r, p=2, dim=1)
        return r  # [num_relations, hidden_channels]

    def _distmult_score(self, u_idx, v_idx, r_idx):
        """Tính DistMult score và chuyển về probability [0-1] bằng sigmoid."""
        hu = self.h_all[u_idx]    # [hidden]
        hv = self.h_all[v_idx]    # [hidden]
        re = self.r_embs[r_idx]   # [hidden]
        # DistMult: score = sum(hu * re * hv)
        # Vì tất cả đã L2-normalized, score nằm trong [-1, 1]
        score = torch.dot(hu * re, hv)
        return score.item()

    def _score_to_prob(self, score):
        """Chuyển DistMult score (phạm vi [-1,1] sau normalize) sang prob [0,1]."""
        # Với L2-normalized embeddings, dùng sigmoid với temperature nhỏ sẽ cho
        # phân phối hợp lý hơn. Scale bằng hidden_dim để amplify signal.
        hidden_dim = self.h_all.shape[1]
        temperature = 1.0 / math.sqrt(hidden_dim)  # = 0.125 with hidden=64
        # tanh maps [-1,1] -> (-1,1), rồi shift về [0,1]
        prob = (math.tanh(score / temperature) + 1) / 2
        return prob

    # ============================================================
    # Public predict methods
    # ============================================================

    def get_prob(self, drug1_smiles, drug2_smiles, side_name):
        """Trả về xác suất (0-1) của cặp thuốc với 1 tác dụng phụ cụ thể"""
        u_idx = self.drug_to_id.get(drug1_smiles)
        v_idx = self.drug_to_id.get(drug2_smiles)
        r_idx = self.side_to_id.get(side_name)
        if u_idx is None or v_idx is None or r_idx is None:
            return None
        with torch.no_grad():
            score = self._distmult_score(u_idx, v_idx, r_idx)
        return self._score_to_prob(score)

    def get_all_side_probs(self, drug1_smiles, drug2_smiles, exclude_sides=None):
        """Tính xác suất tất cả side effects cho một cặp thuốc.
        Trả về list[(prob, side_name)] đã sort giảm dần."""
        u_idx = self.drug_to_id.get(drug1_smiles)
        v_idx = self.drug_to_id.get(drug2_smiles)
        if u_idx is None or v_idx is None:
            return []
        exclude_sides = exclude_sides or set()

        hu = self.h_all[u_idx]
        hv = self.h_all[v_idx]

        results = []
        with torch.no_grad():
            # Vectorized: compute all relation scores at once
            # r_embs: [num_rel, hidden], hu*hv: [hidden]
            huv = hu * hv  # [hidden]
            # scores[r] = sum(hu * r_embs[r] * hv) = dot(huv, r_embs[r].T)
            scores = torch.mv(self.r_embs, huv)  # [num_rel]
            scores_np = scores.cpu().numpy()

        for r_idx, score in enumerate(scores_np):
            side_name = self.id_to_side[r_idx]
            if side_name in exclude_sides:
                continue
            prob = self._score_to_prob(float(score)) * 100
            results.append((prob, side_name))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def get_top_pairs_for_side(self, side_name, drug_smiles,
                               exclude_drugs=None, top_n=10):
        """Với 1 thuốc + 1 tác dụng phụ, tìm top_n thuốc kết hợp tiềm ẩn qua AI"""
        u_idx = self.drug_to_id.get(drug_smiles)
        r_idx = self.side_to_id.get(side_name)
        if u_idx is None or r_idx is None:
            return []
        exclude_drugs = set(exclude_drugs or [])

        hu = self.h_all[u_idx]    # [hidden]
        re = self.r_embs[r_idx]   # [hidden]

        results = []
        with torch.no_grad():
            # hur = hu * re  [hidden]
            hur = hu * re
            # scores for all other drugs: dot(hur, h_all[v].T)
            scores = torch.mv(self.h_all, hur)  # [num_nodes]
            scores_np = scores.cpu().numpy()

        for other_smiles, v_idx in self.drug_to_id.items():
            if other_smiles == drug_smiles or other_smiles in exclude_drugs:
                continue
            prob = self._score_to_prob(float(scores_np[v_idx])) * 100
            results.append((prob, other_smiles))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    def get_top_unknown_pairs_for_side(self, side_name, top_n=20):
        """Top N cặp thuốc chưa có trong dataset có khả năng cao gây side_name"""
        r_idx = self.side_to_id.get(side_name)
        if r_idx is None:
            return []
        existing = set(
            zip(self.df_full['SMILES_1'], self.df_full['SMILES_2'])
        )
        all_drugs = list(self.drug_to_id.keys())
        re = self.r_embs[r_idx]  # [hidden]

        results = []
        with torch.no_grad():
            # Pre-compute h_all * re for all nodes
            h_re = self.h_all * re.unsqueeze(0)  # [num_nodes, hidden]
            # For each pair (i,j): score = dot(h_re[i], h_all[j])
            # score_matrix = h_re @ h_all.T  [num_nodes, num_nodes]
            score_matrix = torch.mm(h_re, self.h_all.t())  # [N, N]

        score_np = score_matrix.cpu().numpy()
        for i, d1 in enumerate(all_drugs):
            for j, d2 in enumerate(all_drugs):
                if j <= i:
                    continue
                if (d1, d2) in existing or (d2, d1) in existing:
                    continue
                u = self.drug_to_id[d1]
                v = self.drug_to_id[d2]
                prob = self._score_to_prob(float(score_np[u][v])) * 100
                results.append((prob, d1, d2))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]