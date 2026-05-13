# ============================================
# Mapping Service — khớp với codeapp.py
# ============================================

import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class MappingService:

    def __init__(self):
        metadata_path = os.path.join(BASE_DIR, "data/mapping/metadata.pkl")
        training_data_path = os.path.join(BASE_DIR, "data/processed/ready_to_train.csv")

        if os.path.exists(metadata_path):
            # --- ƯU TIÊN LOAD TỪ METADATA (CỰC NHANH) ---
            import pickle
            with open(metadata_path, 'rb') as f:
                meta = pickle.load(f)
            
            self.drug_to_id = meta['drug_to_id']
            self.side_to_id = meta['side_to_id']
            self.drug_to_name = meta['drug_to_name']
            self.side_to_vn = meta['side_to_vn']
            self._drug_list = ["Trống"] + meta['all_drugs']
            
            raw_sides = meta['raw_sides']
            self._side_options = ["Tất cả"] + sorted(
                raw_sides, key=lambda x: self.side_to_vn.get(x, x).lower()
            )
            
            self.num_nodes = meta['num_nodes']
            self.num_relations = meta['num_relations']
            self.df = pd.DataFrame() # Để trống hoặc load header nếu cần
            print("✅ Loaded mapping from metadata.pkl")
        else:
            # --- FALLBACK: LOAD TỪ CSV (CHẬM, DÙNG KHI CHƯA CÓ PKL) ---
            print("⚠️ Metadata.pkl not found, loading from CSV (this will be slow)...")
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(f"Training data không tìm thấy: {training_data_path}")

            df = pd.read_csv(training_data_path)
            self.df = df
            all_drugs = sorted(pd.concat([df['SMILES_1'], df['SMILES_2']]).unique())
            self.drug_to_id = {smiles: i for i, smiles in enumerate(all_drugs)}
            raw_sides = sorted(df['Side_Name'].unique())
            self.side_to_id = {name: i for i, name in enumerate(raw_sides)}
            self.num_nodes = len(all_drugs)
            self.num_relations = len(raw_sides)

            # Drug name & Side VN mapping
            self.drug_to_name = {}
            drug_mapping_path = os.path.join(BASE_DIR, "data/mapping/drug_mapping.csv")
            if os.path.exists(drug_mapping_path):
                dm = pd.read_csv(drug_mapping_path)
                self.drug_to_name = dict(zip(dm['SMILES'], dm['Drug_Name']))

            self.side_to_vn = {}
            side_mapping_path = os.path.join(BASE_DIR, "data/mapping/side_effects_mapping.csv")
            if os.path.exists(side_mapping_path):
                sm = pd.read_csv(side_mapping_path)
                self.side_to_vn = dict(zip(sm['Side_Name'], sm['Side_VN']))

            self._drug_list = ["Trống"] + all_drugs
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