import pandas as pd
import pubchempy as pcp
import time
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RAW_DATA_PATH = 'data/raw/ChChSe-Decagon_polypharmacy.csv'
OUTPUT_MAPPING = 'data/mapping/full_cid_to_smiles.csv'

def get_unique_cids(file_path):
    print(f"--- 📂 1. ĐANG QUÉT FILE GỐC: {file_path} ---")
    if not os.path.exists(file_path): return []
    df = pd.read_csv(file_path, sep=',', skiprows=1, names=['Drug1', 'Drug2', 'Side_ID', 'Side_Name'])
    return pd.concat([df['Drug1'], df['Drug2']]).unique()

def fetch_smiles_from_pubchem(cid_list, delay=0.3):
    mapping = []
    total = len(cid_list)
    if total == 0:
        print("✅ Không có thuốc nào mới cần tải.")
        return pd.DataFrame()

    print(f"🚀 2. BẮT ĐẦU TRA CỨU {total} THUỐC CÒN THIẾU...")
    for i, cid in enumerate(cid_list):
        try:
            cid_num = int(''.join(filter(str.isdigit, cid)))
            cpd = pcp.Compound.from_cid(cid_num)
            smiles = cpd.isomeric_smiles if cpd.isomeric_smiles else cpd.canonical_smiles
            mapping.append({'Drug_ID': cid, 'SMILES': smiles})
            print(f"✅ Đã tải: {cid}")
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Lỗi tại {cid}: {e}")
            mapping.append({'Drug_ID': cid, 'SMILES': None})
    return pd.DataFrame(mapping)

if __name__ == "__main__":
    # 1. Lấy danh sách tổng từ file gốc
    all_target_cids = get_unique_cids(RAW_DATA_PATH)
    
    # 2. KIỂM TRA FILE ĐÃ TỒN TẠI (Đoạn này phải nằm ở đây)
    final_cids_to_fetch = all_target_cids
    existing_df = pd.DataFrame(columns=['Drug_ID', 'SMILES'])

    if os.path.exists(OUTPUT_MAPPING):
        existing_df = pd.read_csv(OUTPUT_MAPPING)
        # Lọc bỏ những dòng bị rỗng (SMILES is None/NaN)
        existing_df = existing_df.dropna(subset=['SMILES'])
        existing_cids = set(existing_df['Drug_ID'].unique())
        
        print(f"♻️ Đã có {len(existing_cids)} thuốc trong file.")
        final_cids_to_fetch = [cid for cid in all_target_cids if cid not in existing_cids]

    # 3. Chỉ tải những cái còn thiếu
    if len(final_cids_to_fetch) > 0:
        new_mapping_df = fetch_smiles_from_pubchem(final_cids_to_fetch)
        
        # 4. GỘP CŨ VÀ MỚI RỒI LƯU
        full_df = pd.concat([existing_df, new_mapping_df]).drop_duplicates(subset=['Drug_ID'])
        os.makedirs(os.path.dirname(OUTPUT_MAPPING), exist_ok=True)
        full_df.to_csv(OUTPUT_MAPPING, index=False)
        print(f"✨ Đã cập nhật xong! Tổng cộng: {len(full_df)} thuốc.")
    else:
        print("💎 Dữ liệu đã đầy đủ 100%, không cần tải thêm!")