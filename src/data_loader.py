import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
import pandas as pd

def smiles_to_node_features(smiles):
    """Biến SMILES thành vector đặc trưng của các nguyên tử (Dùng RDKit)"""
    mol = Chem.MolFromSmiles(smiles)
    # Ở đây ta lấy ví dụ đơn giản: số hiệu nguyên tử
    # Trong thực tế, DeepChem sẽ làm phần này xịn hơn
    features = []
    for atom in mol.GetAtoms():
        features.append([atom.GetAtomicNum()])
    return torch.tensor(features, dtype=torch.float)

def create_pyg_graph(df):
    """Biến DataFrame thành đối tượng Data của PyTorch Geometric"""
    # 1. Mã hóa Side_Name thành số (0, 1, 2...)
    df['Side_Label'] = df['Side_Name'].astype('category').cat.codes
    
    # 2. Tạo danh sách các thuốc duy nhất để làm Node
    unique_drugs = pd.concat([df['Drug1'], df['Drug2']]).unique()
    # use uniform name drug_to_id
    drug_to_id = {drug: i for i, drug in enumerate(unique_drugs)}
    
    # 3. Tạo các cạnh (edges) và loại cạnh (edge_types)
    edge_index = []
    edge_type = []
    
    for _, row in df.iterrows():
        u = drug_to_id[row['Drug1']]
        v = drug_to_id[row['Drug2']]
        edge_index.append([u, v])
        edge_type.append(row['Side_Label'])
        
    return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_type)