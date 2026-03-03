# ============================================
# Advanced Inference Engine with Risk Intelligence
# ============================================

import torch
import pandas as pd
import os
from .risk_assessor import RiskAssessor
from .explainer import DDIExplainer


class InferenceEngine:

    def __init__(self, model, drug_map, side_map, device, side_to_vn=None):
        self.model = model
        self.drug_map = drug_map  # SMILES -> node_id
        self.side_map = side_map  # side_name -> side_id
        self.device = device

        self.risk_assessor = RiskAssessor()
        self.explainer = DDIExplainer()

        # Create reverse mapping: side_id -> side_name
        self.id_to_side = {v: k for k, v in side_map.items()}
        
        # side_to_vn mapping for Vietnamese side effect names
        self.side_to_vn = side_to_vn or {}
        
        # Tạo edge_index và edge_type từ training data
        self.edge_index, self.edge_type = self._load_graph_structure()

        self.model.eval()
    
    def _load_graph_structure(self):
        """Tạo edge_index và edge_type từ training data"""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'ready_to_train.csv'))
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Map drug pairs using SMILES -> node_id
        edge_src = [self.drug_map[smiles] for smiles in df['SMILES_1']]
        edge_dst = [self.drug_map[smiles] for smiles in df['SMILES_2']]
        
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(self.device)
        
        # Map side effects using side_name -> side_id
        edge_type = torch.tensor([self.side_map[side_name] for side_name in df['Side_Name']], dtype=torch.long).to(self.device)
        
        return edge_index, edge_type

    def predict_pair(self, drug1: str, drug2: str):
        """
        Dự đoán tương tác dược phẩm giữa hai thuốc
        drug1, drug2: SMILES strings
        """
        
        if drug1 not in self.drug_map or drug2 not in self.drug_map:
            return {
                "error": "One or both drugs not found in mapping."
            }

        d1 = self.drug_map[drug1]
        d2 = self.drug_map[drug2]

        with torch.no_grad():

            # Sử dụng edge_index và edge_type từ training data
            h_emb = self.model(self.edge_index, self.edge_type)

            h = h_emb[d1]
            t = h_emb[d2]

            predictions = []

            for rel_id in range(len(self.side_map)):
                r = self.model.rel_emb.weight[rel_id]
                score = torch.sum(h * r * t)
                prob = torch.sigmoid(score).item()
                
                # Get side effect name from rel_id
                side_name = self.id_to_side.get(rel_id, f"Side_Effect_{rel_id}")

                alert = self.risk_assessor.generate_alert(
                    probability=prob,
                    side_effect=side_name
                )

                predictions.append(alert)

        # Sort highest risk first
        predictions = sorted(
            predictions,
            key=lambda x: x["probability"],
            reverse=True
        )

        # Aggregate analysis
        aggregate = self.explainer.aggregate_pair_risk(predictions)
        explanation = self.explainer.generate_explanation(predictions[:10])

        overall_risk = self._determine_overall_risk(aggregate)

        return {
            "drug_1": drug1,
            "drug_2": drug2,
            "overall_risk_level": overall_risk,
            "summary_analysis": aggregate,
            "clinical_summary": explanation,
            "top_10_risks": predictions[:10],
            "model_generated": True
        }

    def _determine_overall_risk(self, aggregate_data):

        if aggregate_data["risk_distribution"]["CRITICAL"] > 0:
            return "CRITICAL"
        elif aggregate_data["risk_distribution"]["HIGH"] > 5:
            return "HIGH"
        elif aggregate_data["risk_distribution"]["MODERATE"] > 20:
            return "MODERATE"
        else:
            return "LOW"