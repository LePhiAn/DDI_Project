# ============================================
# Mapping Service
# ============================================

import pandas as pd
import sys
import os

# Set up path to import from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class MappingService:

    def __init__(self):
        # Use absolute paths
        drug_mapping_path = os.path.join(BASE_DIR, "data/mapping/drug_mapping.csv")
        side_mapping_path = os.path.join(BASE_DIR, "data/mapping/side_effects_mapping.csv")
        training_data_path = os.path.join(BASE_DIR, "data/processed/ready_to_train.csv")
        
        self.drug_mapping = pd.read_csv(drug_mapping_path)
        self.side_mapping = pd.read_csv(side_mapping_path)

        # Load training data to create proper node ID mappings
        if os.path.exists(training_data_path):
            training_df = pd.read_csv(training_data_path)
            # Get unique drugs from training data
            unique_smiles = pd.concat([training_df['SMILES_1'], training_df['SMILES_2']]).unique()
            
            # Create SMILES -> node_id mapping (integer index)
            self.drug_to_id = {smiles: i for i, smiles in enumerate(unique_smiles)}
            
            # Create SMILES -> drug_name mapping for display (for UI/display purposes)
            self.drug_to_name = dict(
                zip(self.drug_mapping["SMILES"], self.drug_mapping["Drug_Name"])
            )
            
            # Create side_name -> side_id mapping
            unique_sides = training_df['Side_Name'].unique()
            self.side_to_id = {side: i for i, side in enumerate(unique_sides)}
            
            # Create side_name -> side_name_vn mapping
            self.side_to_vn = dict(
                zip(self.side_mapping["Side_Name"], self.side_mapping["Side_VN"])
            )
            
            self.num_nodes = len(unique_smiles)
            self.num_relations = len(unique_sides)
        else:
            raise FileNotFoundError(f"Training data not found at {training_data_path}")

    def get_drug_list(self):
        """Return list of drug names (SMILES) for UI selection"""
        return list(self.drug_to_id.keys())

    def get_maps(self):
        """Return SMILES->node_id and side_name->side_id mappings"""
        return self.drug_to_id, self.side_to_id