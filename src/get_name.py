import pandas as pd
import os

def generate_side_effect_template():
    # 1. Đường dẫn tới file dữ liệu gốc của bạn
    input_path = 'data/processed/ready_to_train.csv'
    # 2. Đường dẫn file mapping sẽ tạo ra
    output_path = 'data/mapping/side_effects_mapping1.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file gốc tại: {input_path}")
        return

    # Đọc dữ liệu
    print("⏳ Đang đọc dữ liệu và trích xuất triệu chứng...")
    df = pd.read_csv(input_path)
    
    # Lấy danh sách các triệu chứng duy nhất và sắp xếp theo bảng chữ cái
    unique_sides = sorted(df['Side_Name'].unique())
    
    # Tạo DataFrame mới để làm mapping
    # Cột Side_VN tạm thời để giống Side_Name để bạn dễ tra cứu khi dịch
    mapping_df = pd.DataFrame({
        'Side_Name': unique_sides,
        'Side_VN': unique_sides 
    })
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lưu file CSV (Dùng encoding utf-8-sig để Excel không bị lỗi font khi bạn gõ Tiếng Việt)
    mapping_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã tạo xong file template tại: {output_path}")
    print(f"📊 Tổng cộng có {len(unique_sides)} triệu chứng cần dịch.")
    print("👉 Giờ bạn hãy mở file này bằng Excel, giữ nguyên cột A, và sửa cột B thành Tiếng Việt nhé!")

if __name__ == "__main__":
    generate_side_effect_template()