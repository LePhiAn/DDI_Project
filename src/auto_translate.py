import pandas as pd
from deep_translator import GoogleTranslator
import os
import time

def auto_translate_with_checkpoint():
    input_path = 'data/processed/ready_to_train.csv'
    output_path = 'data/mapping/side_effects_mapping.csv'
    
    # 1. Lấy danh sách gốc từ file train
    df_raw = pd.read_csv(input_path)
    unique_sides = sorted(df_raw['Side_Name'].unique())
    
    # 2. Kiểm tra nếu đã có file dịch dở (Checkpoint)
    if os.path.exists(output_path):
        mapping_df = pd.read_csv(output_path)
        # Đảm bảo có đủ các triệu chứng mới nếu file train thay đổi
        existing_sides = mapping_df['Side_Name'].tolist()
        new_sides = [s for s in unique_sides if s not in existing_sides]
        if new_sides:
            temp_df = pd.DataFrame({'Side_Name': new_sides, 'Side_VN': ''})
            mapping_df = pd.concat([mapping_df, temp_df], ignore_index=True)
    else:
        mapping_df = pd.DataFrame({'Side_Name': unique_sides, 'Side_VN': ''})

    # 3. Cấu hình bộ dịch
    translator = GoogleTranslator(source='en', target='vi')
    
    print(f"🔄 Tổng số: {len(mapping_df)} dòng.")
    print("🚀 Bắt đầu dịch (Dịch tới đâu lưu tới đó)...")

    for index, row in mapping_df.iterrows():
        # Chỉ dịch nếu cột Side_VN đang trống hoặc bằng chính Side_Name
        if pd.isna(row['Side_VN']) or row['Side_VN'] == "" or row['Side_VN'] == row['Side_Name']:
            try:
                english_text = str(row['Side_Name'])
                vietnamese_text = translator.translate(english_text)
                
                # Cập nhật vào DataFrame
                mapping_df.at[index, 'Side_VN'] = vietnamese_text
                
                # LƯU NGAY LẬP TỨC VÀO FILE
                mapping_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                print(f"✅ [{index+1}/{len(mapping_df)}] {english_text} -> {vietnamese_text}")
                
                # Nghỉ 0.2s để tránh bị Google chặn vì gửi yêu cầu quá nhanh
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Lỗi tại dòng {index+1}: {e}. Đang tạm dừng 5s...")
                time.sleep(5)
                continue
        else:
            # Nếu dòng này đã có tiếng Việt rồi thì bỏ qua
            pass

    print(f"✨ HOÀN THÀNH! File lưu tại: {output_path}")

if __name__ == "__main__":
    auto_translate_with_checkpoint()