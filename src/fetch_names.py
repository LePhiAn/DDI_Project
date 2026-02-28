import pubchempy as pcp
import pandas as pd
import os
import time

def fetch_drug_names():
    input_path = 'data/processed/ready_to_train.csv'
    output_path = 'data/mapping/drug_mapping.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file đầu vào tại: {input_path}")
        return

    # 1. Đọc dữ liệu SMILES cần tra cứu
    df = pd.read_csv(input_path)
    all_smiles = pd.concat([df['SMILES_1'], df['SMILES_2']]).unique()
    
    # 2. Đọc dữ liệu đã có (để bỏ qua nếu chạy lại)
    existing_smiles = set()
    if os.path.exists(output_path):
        try:
            old_df = pd.read_csv(output_path)
            existing_smiles = set(old_df['SMILES'].tolist())
            print(f"♻️ Đã tìm thấy file cũ. Sẽ bỏ qua {len(existing_smiles)} thuốc đã có tên.")
        except:
            print("⚠️ File cũ bị lỗi, sẽ ghi đè mới.")

    # Lọc ra những SMILES chưa có tên
    test_smiles = [s for s in all_smiles if s not in existing_smiles]
    
    if not test_smiles:
        print("✅ Tất cả thuốc đã được tra cứu xong!")
        return

    print(f"🚀 Còn lại {len(test_smiles)}/{len(all_smiles)} thuốc cần tra cứu...")

    for i, smi in enumerate(test_smiles):
        try:
            # Tra cứu PubChem
            compounds = pcp.get_compounds(smi, namespace='smiles')
            
            name = "Unknown"
            if compounds and len(compounds) > 0:
                name = compounds[0].synonyms[0] if compounds[0].synonyms else "Unknown"
            
            # Tạo DataFrame dòng mới
            new_row = pd.DataFrame([{"SMILES": smi, "Drug_Name": name}])
            
            # LƯU NGAY LẬP TỨC (Mode 'a' - append)
            # Nếu file chưa có thì ghi cả Header, có rồi thì chỉ ghi dữ liệu
            header = not os.path.exists(output_path)
            new_row.to_csv(output_path, mode='a', index=False, header=header)
            
            print(f"[{i+1}/{len(test_smiles)}] ✅ Đã lưu: {name}")
            
            # Nghỉ một chút tránh bị chặn
            time.sleep(0.3) 
            
        except Exception as e:
            # Nếu lỗi mạng (Timeout), nghỉ 2 giây rồi tiếp tục dòng sau
            if "10060" in str(e) or "timeout" in str(e).lower():
                print(f"❌ Lỗi mạng tại dòng {i+1}. Đang tạm nghỉ 2s...")
                time.sleep(2)
            else:
                print(f"❌ Lỗi: {str(e)}")
            continue

    print("\n" + "="*30)
    print(f"🎉 HOÀN THÀNH! Dữ liệu đã được cập nhật tại: {output_path}")
    print("="*30)

if __name__ == "__main__":
    fetch_drug_names()