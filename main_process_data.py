import pandas as pd
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RAW_DATA_PATH = 'data/raw/ChChSe-Decagon_polypharmacy.csv'
MAPPING_PATH = 'data/mapping/full_cid_to_smiles.csv'
OUTPUT_PATH = 'data/processed/ready_to_train.csv'

# 1. Load dữ liệu thô
print("--- 📂 1. ĐANG ĐỌC DỮ LIỆU THÔ ---")
df_raw = pd.read_csv(RAW_DATA_PATH, skiprows=1, 
                     names=['Drug1', 'Drug2', 'Side_ID', 'Side_Name'])

df_mapping = pd.read_csv(MAPPING_PATH)

# --- BƯỚC KIỂM TRA (THỐNG KÊ LÝ DO MẤT DỮ LIỆU) ---
print("--- 📊 2. THỐNG KÊ TRƯỚC KHI MERGE ---")
drugs_in_raw = pd.concat([df_raw['Drug1'], df_raw['Drug2']]).unique()
drugs_in_mapping = df_mapping['Drug_ID'].unique()

missing_drugs = set(drugs_in_raw) - set(drugs_in_mapping)

print(f"  + Tổng số thuốc trong file Decagon: {len(drugs_in_raw)}")
print(f"  + Tổng số thuốc có SMILES mapping: {len(drugs_in_mapping)}")
print(f"  + Số lượng thuốc THIẾU cấu trúc hóa học: {len(missing_drugs)}")
print(f"  + Tỉ lệ thuốc giữ lại được: {((len(drugs_in_raw)-len(missing_drugs))/len(drugs_in_raw))*100:.2f}%")
print("-" * 40)

# 2. Merge SMILES cho thuốc thứ nhất (Drug1)
print("--- 🔗 3. ĐANG ÁNH XẠ SMILES CHO DRUG 1 ---")
df = pd.merge(df_raw, df_mapping, left_on='Drug1', right_on='Drug_ID').drop('Drug_ID', axis=1)
df = df.rename(columns={'SMILES': 'SMILES_1'})

# 3. Merge SMILES cho thuốc thứ hai (Drug2)
print("--- 🔗 4. ĐANG ÁNH XẠ SMILES CHO DRUG 2 ---")
df = pd.merge(df, df_mapping, left_on='Drug2', right_on='Drug_ID').drop('Drug_ID', axis=1)
df = df.rename(columns={'SMILES': 'SMILES_2'})

# 4. Làm sạch dữ liệu (Xóa bỏ các dòng thiếu thông tin)
df_final = df.dropna(subset=['SMILES_1', 'SMILES_2'])

# Loại bỏ trùng lặp (nếu có)
df_final = df_final.drop_duplicates()

# 5. Lưu kết quả cuối cùng
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"--- 💾 5. ĐANG LƯU KẾT QUẢ ({len(df_final)} dòng) ---")
df_final.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*40)
print(f"✅ HOÀN THÀNH XỬ LÝ DỮ LIỆU!")
print(f"  - Tổng số cặp tương tác thu được: {len(df_final)}")
print(f"  - Số lượng loại tác dụng phụ (Relations): {df_final['Side_Name'].nunique()}")
print(f"  - File đã lưu tại: {OUTPUT_PATH}")
print("="*40)