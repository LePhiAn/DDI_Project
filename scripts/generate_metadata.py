import pandas as pd
import pickle
import os
import sys

# Thêm root vào sys.path để import các service nếu cần
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

def generate_metadata():
    print("--- DANG TRICH XUAT METADATA TU FILE CSV 688MB ---")

    
    training_data_path = os.path.join(ROOT_DIR, "data/processed/ready_to_train.csv")
    drug_mapping_path  = os.path.join(ROOT_DIR, "data/mapping/drug_mapping.csv")
    side_mapping_path  = os.path.join(ROOT_DIR, "data/mapping/side_effects_mapping.csv")
    output_path = os.path.join(ROOT_DIR, "data/mapping/metadata.pkl")

    if not os.path.exists(training_data_path):
        print(f"❌ Không tìm thấy file: {training_data_path}")
        return

    # 1. Đọc dữ liệu (chỉ lấy các cột cần thiết để tiết kiệm RAM lúc xử lý)
    df = pd.read_csv(training_data_path, usecols=['SMILES_1', 'SMILES_2', 'Side_Name'])
    
    # 2. Tạo Mapping Drug -> ID
    all_drugs = sorted(pd.concat([df['SMILES_1'], df['SMILES_2']]).unique())
    drug_to_id = {smiles: i for i, smiles in enumerate(all_drugs)}
    
    # 3. Tạo Mapping Side -> ID
    raw_sides = sorted(df['Side_Name'].unique())
    side_to_id = {name: i for i, name in enumerate(raw_sides)}
    
    # 4. Load tên thuốc và dịch tiếng Việt (nếu có)
    drug_to_name = {}
    if os.path.exists(drug_mapping_path):
        dm = pd.read_csv(drug_mapping_path)
        drug_to_name = dict(zip(dm['SMILES'], dm['Drug_Name']))

    side_to_vn = {}
    if os.path.exists(side_mapping_path):
        sm = pd.read_csv(side_mapping_path)
        side_to_vn = dict(zip(sm['Side_Name'], sm['Side_VN']))

    # 5. Đóng gói dữ liệu
    metadata = {
        'drug_to_id': drug_to_id,
        'side_to_id': side_to_id,
        'drug_to_name': drug_to_name,
        'side_to_vn': side_to_vn,
        'all_drugs': all_drugs,
        'raw_sides': raw_sides,
        'num_nodes': len(all_drugs),
        'num_relations': len(raw_sides)
    }

    # 6. Lưu file Pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"DONE: {output_path}")
    print(f"Stats: {len(all_drugs)} drugs, {len(raw_sides)} side effects.")
    print(f"Size: {os.path.getsize(output_path) / 1024:.2f} KB (Vs 688 MB CSV)")

if __name__ == "__main__":
    generate_metadata()
