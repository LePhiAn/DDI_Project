# ============================================
# Mapping Service — khớp với codeapp.py
# ============================================

import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class MappingService:

    def __init__(self):
        training_data_path = os.path.join(BASE_DIR, "data/processed/ready_to_train.csv")
        drug_mapping_path  = os.path.join(BASE_DIR, "data/mapping/drug_mapping.csv")
        side_mapping_path  = os.path.join(BASE_DIR, "data/mapping/side_effects_mapping.csv")

        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"Training data không tìm thấy: {training_data_path}")

        df = pd.read_csv(training_data_path)
        self.df = df  # giữ để dùng bên ngoài

        # Tất cả drugs theo thứ tự sorted (giống codeapp.py)
        all_drugs = sorted(pd.concat([df['SMILES_1'], df['SMILES_2']]).unique())
        self.drug_to_id = {smiles: i for i, smiles in enumerate(all_drugs)}

        # Side effects
        raw_sides = sorted(df['Side_Name'].unique())
        self.side_to_id = {name: i for i, name in enumerate(raw_sides)}
        self.raw_sides = raw_sides

        # Số lượng
        self.num_nodes = len(all_drugs)
        self.num_relations = len(raw_sides)

        # Drug name display mapping
        self.drug_to_name = {}
        if os.path.exists(drug_mapping_path):
            dm = pd.read_csv(drug_mapping_path)
            self.drug_to_name = dict(zip(dm['SMILES'], dm['Drug_Name']))

        # Side effect VN mapping
        self.side_to_vn = {}
        if os.path.exists(side_mapping_path):
            sm = pd.read_csv(side_mapping_path)
            self.side_to_vn = dict(zip(sm['Side_Name'], sm['Side_VN']))

        # Drug list: ["Trống"] + all_drugs (dùng cho selectbox)
        self._drug_list = ["Trống"] + all_drugs

        # Side options: ["Tất cả"] + sorted by VN name (dùng cho selectbox)
        self._side_options = ["Tất cả"] + sorted(
            raw_sides, key=lambda x: self.side_to_vn.get(x, x).lower()
        )

    # ---- Public API ----

    def get_drug_list(self):
        """["Trống"] + tất cả SMILES để dùng selectbox"""
        return self._drug_list

    def get_side_options(self):
        """["Tất cả"] + tên tác dụng phụ sorted theo VN"""
        return self._side_options

    def get_maps(self):
        """drug_to_id, side_to_id"""
        return self.drug_to_id, self.side_to_id

    def get_display_name(self, smiles):
        """Trả về tên hiển thị đẹp cho một SMILES"""
        if smiles == "Trống":
            return "Trống"
        name = str(self.drug_to_name.get(smiles, "Unknown"))
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        try:
            mol = Chem.MolFromSmiles(smiles)
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol else smiles
        except Exception:
            formula = smiles
        if (name == "Unknown" or name.isdigit() or "-" in name
                or "DTXSID" in name or len(name) > 30):
            return formula
        return name