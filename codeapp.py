import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import os

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="AI Pharma - Drug Interaction Insights", layout="wide")

# CHÈN ĐOẠN CSS CỦA BẠN VÀO ĐÂY ĐỂ "KHÓA" CON TRỎ CHUỘT
st.markdown("""
    <style>
    /* Khóa con trỏ chuột và chặn việc chọn văn bản ở mọi nơi */
    html, body, [data-testid="stAppViewContainer"] {
        cursor: default;
        user-select: none;
    }

    /* CHỈ CHO PHÉP 3 Widget cụ thể được hiện dấu gõ (Selectbox) */
    div[data-testid="stSelectbox"] input {
        cursor: text !important;
        user-select: auto !important;
    }

    /* Đảm bảo các tiêu đề và văn bản khác không hiện dấu nháy */
    h1, h2, h3, h4, h5, h6, p, span, div {
        -webkit-user-select: none;
        user-select: none;
    }
    
    /* Cho phép các ô nhập liệu hoạt động bình thường nếu cần */
    input, textarea {
        cursor: text !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Định nghĩa Model R-GCN – dùng cùng kiến trúc như lúc train (RGCNPredictor)
class RGCNPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super(RGCNPredictor, self).__init__()
        from torch_geometric.nn import RGCNConv
        # embedding thuốc (16‑dim) + hai lớp RGCN
        self.node_emb = nn.Embedding(num_nodes, 16)
        self.conv1 = RGCNConv(16, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        # embedding quan hệ
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

    def forward(self, edge_index, edge_type):
        # trả về biểu diễn nút theo graph (edge_index/edge_type có thể là toàn bộ hoặc chỉ cặp)
        x = self.node_emb.weight
        h = self.conv1(x, edge_index, edge_type).relu()
        h = self.conv2(h, edge_index, edge_type)
        return h

# ==========================================
# 2. HÀM HỖ TRỢ (HELPER FUNCTIONS)
# ==========================================
def get_mol_formula(smiles):
    """Chuyển SMILES thành công thức rút gọn như C6H8O6"""
    if smiles == "Trống":
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return rdMolDescriptors.CalcMolFormula(mol)
        return smiles
    except:
        return smiles
    
def get_display_name(smiles, mapping_dict):
    if smiles == "Trống": return smiles
    
    # Lấy tên từ file CSV, nếu không có thì để "Unknown"
    name = str(mapping_dict.get(smiles, "Unknown"))
    
    # Tính công thức hóa học (C6H8O6...) làm phương án dự phòng
    formula = get_mol_formula(smiles) 
    
    # Logic: Nếu tên là số, có gạch ngang, quá dài (>30 ký tự) hoặc là "Unknown"
    # -> Hiển thị Công thức hóa học cho chuyên nghiệp
    if (name == "Unknown" or name.isdigit() or "-" in name or 
        "DTXSID" in name or len(name) > 30):
        return formula
        
    return name

def plot_drug_risks(smiles, title_name):
    # 1. Lấy dữ liệu rủi ro của thuốc
    stats = df_full[df_full['SMILES_1'] == smiles]['Side_Name'].value_counts().head(5)
    
    if not stats.empty:
        # 2. CHUYỂN ĐỔI NHÃN SANG TIẾNG VIỆT
        # Chúng ta dùng side_vn_map để lấy tên VN, nếu không có thì giữ tên gốc
        vn_labels = [side_vn_map.get(label, label) for label in stats.index]
        
        # 3. VẼ BIỂU ĐỒ
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Đưa vn_labels vào làm nhãn thay cho stats.index
        ax.pie(
            stats, 
            labels=vn_labels, 
            autopct='%1.0f%%', 
            startangle=90, 
            colors=plt.cm.Pastel1.colors, # Dùng hệ màu Pastel cho đẹp giống hình bạn gửi
            textprops={'fontsize': 10}
        )
        
        ax.set_title(f"Thống kê tương tác phổ biến của: {title_name}", fontsize=12, pad=20)
        
        # Hiển thị lên Streamlit
        st.pyplot(fig)
    else:
        st.write("Chưa có dữ liệu thống kê rủi ro cho chất này.")

# ==========================================
# 3. TẢI TÀI NGUYÊN
# ==========================================
@st.cache_resource
def load_resources():
    df = pd.read_csv('data/processed/ready_to_train.csv')
    all_drugs = sorted(pd.concat([df['SMILES_1'], df['SMILES_2']]).unique())
    drug_to_id = {smiles: i for i, smiles in enumerate(all_drugs)}
    raw_side_effects = sorted(df['Side_Name'].unique())
    side_to_id = {name: i for i, name in enumerate(raw_side_effects)}

    # dùng kiến trúc đã huấn luyện
    model = RGCNPredictor(len(all_drugs), len(raw_side_effects), 64)
    model_path = 'models/r_gcn_full_model.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # checkpoint được lưu dưới dạng state_dict thuần
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"⚠️ Lỗi load model: {e}")
    model.eval()

    # xây chỉ số đồ thị đầy đủ cho toàn dataset (dùng cho AI prediction)
    u_indices = df['SMILES_1'].map(drug_to_id).tolist()
    v_indices = df['SMILES_2'].map(drug_to_id).tolist()
    pos_edge_index = torch.tensor([u_indices, v_indices], dtype=torch.long)
    pos_edge_type = torch.tensor(df['Side_Name'].map(side_to_id).tolist(), dtype=torch.long)
    with torch.no_grad():
        h_all = model(pos_edge_index, pos_edge_type)

    path_side_map = 'data/mapping/side_effects_mapping.csv'

    path_side_map = 'data/mapping/side_effects_mapping.csv'
    try:
        side_map_df = pd.read_csv(path_side_map)
        side_vn_map = dict(zip(side_map_df['Side_Name'], side_map_df['Side_VN']))
    except:
        side_vn_map = {s: s for s in raw_side_effects}

    drug_names = {}
    if os.path.exists('data/mapping/drug_mapping.csv'):
        mapping_df = pd.read_csv('data/mapping/drug_mapping.csv')
        drug_names = dict(zip(mapping_df['SMILES'], mapping_df['Drug_Name']))

    side_options = ["Tất cả"] + sorted(raw_side_effects, key=lambda x: side_vn_map.get(x, x).lower())
    return df, ["Trống"] + all_drugs, side_options, drug_names, len(raw_side_effects), side_vn_map, model, drug_to_id, side_to_id, h_all

(df_full, drug_list, side_list, drug_names, num_side_total, side_vn_map, model_ai, drug_to_id, side_to_id, h_all) = load_resources()

if 'd1' not in st.session_state: st.session_state.d1 = "Trống"
if 'd2' not in st.session_state: st.session_state.d2 = "Trống"
if 'side' not in st.session_state: st.session_state.side = "Tất cả"

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.header(" BẢNG ĐIỀU KHIỂN")
    with st.expander("🎲 Thao tác nhanh", expanded=True):
        if st.button("Random cặp có tương tác", use_container_width=True):
            random_row = df_full.sample(n=1).iloc[0]
            st.session_state.d1 = random_row['SMILES_1']
            st.session_state.d2 = random_row['SMILES_2']
            st.session_state.side = random_row['Side_Name']
            st.rerun()

    with st.expander("📊 Thống kê & Báo cáo", expanded=True):
        show_top_10 = st.button("🏆 Top 10 Tác dụng phụ", use_container_width=True)
        show_extreme = st.button("🚨 Top 20 cặp rủi ro cao", width="stretch")

    st.markdown("---")
    st.info(f"""
    📊 **Thống kê dữ liệu:**
    - Tổng số cặp: {len(df_full):,}
    - Số loại triệu chứng: {len(side_list)-1} 
    - Model: R-GCN
    """)

# ==========================================
# 5. GIAO DIỆN CHÍNH (INPUT)
# ==========================================
st.title("💊 Hệ Thống Phân Tích Tương Tác Thuốc")

# Khung điều khiển tập trung
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        d1_select = st.selectbox(
            "💉 Chọn thuốc thứ nhất:", 
            drug_list, 
            index=drug_list.index(st.session_state.d1),
            format_func=lambda x: get_display_name(x, drug_names)
        )
    with col2:
        d2_select = st.selectbox(
            "💉 Chọn thuốc thứ hai:", 
            drug_list,
            index=drug_list.index(st.session_state.d2),
            format_func=lambda x: get_display_name(x, drug_names)
        )

    try: current_side_idx = side_list.index(st.session_state.side)
    except: current_side_idx = 0
    
    selected_side = st.selectbox(
    "⚠️ Chỉ định tác dụng phụ muốn đối chiếu:", 
    side_list, 
    index=current_side_idx,
    format_func=lambda x: side_vn_map.get(x, x) if x != "Tất cả" else "Tất cả rủi ro"
)

    # NÚT BẤM ĐẶT DƯỚI Ô CHỈ ĐỊNH
    btn_analyze = st.button("PHÂN TÍCH", use_container_width=True, type="primary")

# ==========================================
# 6. PHÂN TÍCH (GIỮ NGUYÊN UI - FIX LOGIC THỤT ĐẦU DÒNG)
# ==========================================
if btn_analyze:
    st.markdown("---")
    side_name_vn = side_vn_map.get(selected_side, "Tất cả rủi ro")
    box_style = "text-align: start; background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-inline-start: 5px solid;"
    smiles_style = "word-break: break-all; color: #e83e8c; font-size: 11px; background-color: #f1f1f1; padding: 2px 5px; border-radius: 3px;"

    # TRƯỜNG HỢP 1: CẢ 2 TRỐNG
    if d1_select == "Trống" and d2_select == "Trống":
        if selected_side == "Tất cả":
            # Show danh sách triệu chứng (giống nút sidebar cũ)
            st.markdown("---")
            st.subheader(f"📋 Danh mục rủi ro hệ thống ({num_side_total} loại)")
            clean_sides = [s for s in side_list if s and s != "Tất cả"]
            sides_df = pd.DataFrame({
                "STT": range(1, len(clean_sides) + 1),
                "Tên gốc (EN)": clean_sides,
                "Bản dịch (VN)": [side_vn_map.get(s, s) for s in clean_sides]
            })
            st.dataframe(sides_df, width="stretch", height=400, hide_index=True)
        else:
            # Chỉ định 1 tác dụng phụ - show 2 bảng: Dataset + AI Prediction
            st.subheader(f"🔎 Tác dụng phụ: {side_name_vn}")
            # (reuse same code as BẢNG 1 + BẢNG 2 earlier for side-specific)
            full_pairs = df_full[df_full['Side_Name'] == selected_side]
            if not full_pairs.empty:
                st.markdown("### ✅ Bảng 1: Các cặp ghi nhận trong dataset")
                st.info(f"Tìm thấy **{len(full_pairs):,}** cặp thuốc với tác dụng phụ này.")
                pairs_display = full_pairs.copy()
                pairs_display['Thuốc 1'] = pairs_display['SMILES_1'].apply(lambda x: drug_names.get(x, x))
                pairs_display['Thuốc 2'] = pairs_display['SMILES_2'].apply(lambda x: drug_names.get(x, x))
                with st.expander("📊 Xem bảng toàn bộ cặp"):
                    st.dataframe(
                        pairs_display[['Thuốc 1', 'Thuốc 2']],
                        width="stretch",
                        hide_index=True
                    )
            else:
                st.info("Chưa có cặp thuốc nào với tác dụng phụ này trong dataset.")

            # ========== BẢNG 2: Dự đoán AI ==========
            st.markdown("---")
            st.markdown("### 🤖 Bảng 2: Dự đoán AI - Các cặp tiềm ẩn ngoài dataset")
            st.caption("AI dự báo các cặp thuốc chưa ghi nhận nhưng có khả năng cao gây ra triệu chứng này.")
            r_idx = side_to_id.get(selected_side)
            if r_idx is not None:
                try:
                    predictions = []
                    all_drugs = list(drug_to_id.keys())
                    num_drugs = len(all_drugs)
                    existing_pairs_set = set()
                    for _, row in full_pairs.iterrows():
                        existing_pairs_set.add((row['SMILES_1'], row['SMILES_2']))
                    # pre‑compute relation embedding once
                    r_emb = model_ai.rel_emb(torch.tensor([r_idx], dtype=torch.long))
                    # use h_all (được tạo lúc load_resources) để tính điểm nhanh
                    with torch.no_grad():
                        for i, drug1 in enumerate(all_drugs):
                            for drug2 in all_drugs[i+1:]:
                                if (drug1, drug2) in existing_pairs_set or (drug2, drug1) in existing_pairs_set:
                                    continue
                                u_idx = drug_to_id[drug1]
                                v_idx = drug_to_id[drug2]
                                # compute score using global embeddings
                                score = torch.sum(h_all[u_idx] * r_emb * h_all[v_idx])
                                prob = torch.sigmoid(score).item() * 100
                                predictions.append((prob, drug1, drug2))
                    if predictions:
                        predictions.sort(key=lambda x: x[0], reverse=True)
                        top_predictions = predictions[:20]
                        pred_data = []
                        for prob, drug1, drug2 in top_predictions:
                            pred_data.append({
                                'prob': prob,
                                'Xác suất (%)': f"{prob:.1f}%",
                                'Thuốc 1': drug_names.get(drug1, drug1),
                                'Thuốc 2': drug_names.get(drug2, drug2)
                            })
                        pred_df = pd.DataFrame(pred_data).sort_values('prob', ascending=False).reset_index(drop=True)
                        left_col, right_col = st.columns([1, 1])
                        with left_col:
                            with st.expander("📊 Bảng top 20 dự đoán"):
                                st.dataframe(pred_df[['Xác suất (%)', 'Thuốc 1', 'Thuốc 2']], width="stretch", hide_index=True)
                        with right_col:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            names = pred_df.apply(lambda r: f"{r['Thuốc 1']} | {r['Thuốc 2']}", axis=1)
                            probs = pred_df['prob'].values
                            y_pos = list(range(len(names)))
                            ax.barh(y_pos, probs, color='tab:blue')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(names)
                            ax.invert_yaxis()
                            ax.set_xlabel('Xác suất (%)')
                            ax.set_title('Top 20 cặp dự đoán AI')
                            for i, v in enumerate(probs):
                                ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9)
                            st.pyplot(fig)
                    else:
                        st.info("Không có cặp tiền ẩn được AI dự đoán.")
                except Exception as e:
                    st.error(f"Lỗi khi tính toán AI prediction: {e}")
            else:
                st.warning("Không thể chạy AI dự đoán.")

    # TRƯỜNG HỢP 2: ĐƠN CHẤT
    elif d1_select == "Trống" or d2_select == "Trống":
        active_drug = d1_select if d1_select != "Trống" else d2_select
        short_name = drug_names.get(active_drug, active_drug)
        st.subheader(f"🔍 Phân tích chi tiết: {short_name}")

        if selected_side == "Tất cả":
            # giống phần phân tích chung của 1 thuốc: hình + tên và biểu đồ
            col_visual, col_chart = st.columns([1, 1])
            with col_visual:
                mol = Chem.MolFromSmiles(active_drug)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(350, 350)), width=350)
                formula = get_mol_formula(active_drug)
                st.markdown(f"""
                <div style="{box_style} border-inline-start-color: #9b59b6;">
                    <small><b>Tên gọi:</b> {short_name}</small><br>
                    <small><b>Công thức rút gọn:</b> {formula}</small><br>
                    <small><b>Mã SMILES:</b></small><br>
                    <div style="{smiles_style}">{active_drug}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_chart:
                stats = df_full[df_full['SMILES_1'] == active_drug]['Side_Name'].value_counts().head(10)
                if not stats.empty:
                    vn_labels = [side_vn_map.get(label, label) for label in stats.index]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(
                        stats,
                        labels=vn_labels,
                        autopct='%1.0f%%',
                        startangle=90,
                        colors=plt.cm.Pastel1.colors,
                        textprops={'fontsize': 9}
                    )
                    ax.set_title(f"Tác dụng phụ phổ biến", fontsize=11, fontweight='bold', pad=15)
                    st.pyplot(fig)
                else:
                    st.info("Chưa có dữ liệu thống kê rủi ro cho chất này.")
            # liệt kê tất cả tác dụng phụ của chất này
            st.markdown("---")
            st.subheader("📋 Tất cả tác dụng phụ của chất này")
            all_side_effects = df_full[df_full['SMILES_1'] == active_drug]['Side_Name'].unique()
            if len(all_side_effects) > 0:
                side_effects_vn = [side_vn_map.get(s, s) for s in all_side_effects]
                st.write(f"**Tổng cộng {len(all_side_effects)} tác dụng phụ:**")
                cols = st.columns(4)
                for i, side in enumerate(side_effects_vn):
                    with cols[i % 4]:
                        st.write(f"• {side}")
            else:
                st.info("Chưa có dữ liệu tác dụng phụ cho chất này.")
        else:
            # cụ thể 1 thuốc + 1 tác dụng phụ: left dataset + right AI
            side_name_vn = side_vn_map.get(selected_side, selected_side)
            st.subheader(f"🔍 Phân tích chi tiết: {short_name} & {side_name_vn}")
            existing_pairs = df_full[(df_full['SMILES_1'] == active_drug) & (df_full['Side_Name'] == selected_side)]
            if not existing_pairs.empty:
                    # show drug info and pie chart as earlier
                col_visual, col_chart = st.columns([1, 1])
                with col_visual:
                    mol = Chem.MolFromSmiles(active_drug)
                    if mol:
                        st.image(Draw.MolToImage(mol, size=(350, 350)), width=350)
                    formula = get_mol_formula(active_drug)
                    st.markdown(f"""
                    <div style="{box_style} border-inline-start-color: #9b59b6;">
                        <small><b>Tên gọi:</b> {short_name}</small><br>
                        <small><b>Công thức rút gọn:</b> {formula}</small><br>
                        <small><b>Mã SMILES:</b></small><br>
                        <div style="{smiles_style}">{active_drug}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_chart:
                    stats = df_full[df_full['SMILES_1'] == active_drug]['Side_Name'].value_counts().head(10)
                    if not stats.empty:
                        vn_labels = [side_vn_map.get(label, label) for label in stats.index]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(
                            stats,
                            labels=vn_labels,
                            autopct='%1.0f%%',
                            startangle=90,
                            colors=plt.cm.Pastel1.colors,
                            textprops={'fontsize': 9}
                        )
                        ax.set_title(f"Tác dụng phụ phổ biến", fontsize=11, fontweight='bold', pad=15)
                        st.pyplot(fig)
                    else:
                        st.info("Chưa có dữ liệu thống kê rủi ro cho chất này.")
                # now tables
                st.markdown("---")
                st.markdown(f"#### Danh sách các chất kết hợp với {short_name} gây {side_name_vn}")
                combined_drugs = existing_pairs['SMILES_2'].unique()
                combined_names = [drug_names.get(s, s) for s in combined_drugs]
                left_df = pd.DataFrame({'STT': range(1, len(combined_names)+1), 'Tên thuốc': combined_names})
                # AI side
                r_idx = side_to_id.get(selected_side)
                u_idx = drug_to_id.get(active_drug)
                ai_preds = []
                if model_ai and u_idx is not None and r_idx is not None:
                    try:
                        all_other_drugs = [d for d in drug_to_id.keys() if d != active_drug and d not in combined_drugs]
                        # compute once relation embedding
                        r_emb = model_ai.rel_emb(torch.tensor([r_idx], dtype=torch.long))
                        with torch.no_grad():
                            for other_drug in all_other_drugs:
                                v_idx = drug_to_id[other_drug]
                                score = torch.sum(h_all[u_idx] * r_emb * h_all[v_idx])
                                prob = torch.sigmoid(score).item() * 100
                                ai_preds.append((prob, other_drug))
                        ai_preds.sort(key=lambda x: x[0], reverse=True)
                        top_ai = ai_preds[:10]
                        right_df = pd.DataFrame({
                            'STT': range(1, len(top_ai)+1),
                            'Tên thuốc': [drug_names.get(d, d) for _, d in top_ai],
                            '% xảy ra': [f"{p:.1f}%" for p, _ in top_ai]
                        })
                    except Exception as e:
                        st.error(f"Lỗi khi tính toán dự đoán AI: {e}")
                        right_df = pd.DataFrame(columns=['STT','Tên thuốc','% xảy ra'])
                else:
                    right_df = pd.DataFrame(columns=['STT','Tên thuốc','% xảy ra'])
                lc, rc = st.columns([1,1])
                with lc: st.table(left_df)
                with rc: st.table(right_df)
            else:
                # no existing pairs → still compute predictions for all others
                st.markdown("### Dự đoán các chất kết hợp tiềm ẩn")
                st.info(f"Tác dụng phụ **{side_name_vn}** chưa có dữ liệu ghi nhận. AI sẽ dự đoán những chất nào có khả năng kết hợp với {short_name} gây nên tác dụng này.")
                r_idx = side_to_id.get(selected_side)
                u_idx = drug_to_id.get(active_drug)
                if model_ai and u_idx is not None and r_idx is not None:
                    try:
                        predictions = []
                        all_other_drugs = [d for d in drug_to_id.keys() if d != active_drug]
                        # compute score from cached embeddings
                        r_emb = model_ai.rel_emb(torch.tensor([r_idx], dtype=torch.long))
                        with torch.no_grad():
                            for other_drug in all_other_drugs:
                                v_idx = drug_to_id[other_drug]
                                score = torch.sum(h_all[u_idx] * r_emb * h_all[v_idx])
                                prob = torch.sigmoid(score).item()
                                predictions.append((prob*100, other_drug))
                        predictions.sort(key=lambda x: x[0], reverse=True)
                        top_predictions = predictions[:10]
                        st.markdown("#### Top 10 chất có khả năng cao:")
                        cols = st.columns(3)
                        for i,(prob,drug) in enumerate(top_predictions):
                            drug_name = drug_names.get(drug, drug)
                            with cols[i % 3]:
                                mol = Chem.MolFromSmiles(drug)
                                if mol: st.image(Draw.MolToImage(mol, size=(180,180)))
                                st.write(f"**{drug_name}**")
                                st.write(f"Xác suất: **{prob:.1f}%**")
                                st.caption(f"SMILES: {drug[:25]}..." if len(drug)>25 else f"SMILES: {drug}")
                    except Exception as e:
                        st.error(f"Lỗi khi tính toán dự đoán: {e}")
                else:
                    st.warning("Không thể chạy AI dự đoán. Vui lòng kiểm tra model.")

    # TRƯỜNG HỢP 3: CẶP ĐÔI (CHÚ Ý: ĐÃ ĐƯA VÀO TRONG KHỐI if btn_analyze)
    else:
        if d1_select == d2_select:
            st.error(" Lỗi: Hai loại thuốc chọn không được trùng nhau.")
        else:
            short_name1 = drug_names.get(d1_select, d1_select)
            short_name2 = drug_names.get(d2_select, d2_select)
            st.subheader(f"🧪 Đối chiếu: {short_name1} & {short_name2}")
            
            # Giữ nguyên layout 2 cột ảnh Mol của bạn
            # Giữ nguyên layout 2 cột ảnh Mol nhưng làm mới phần text
            st_c1, st_c2 = st.columns(2)
            
            # Thông tin thuốc 1
            with st_c1:
                mol1 = Chem.MolFromSmiles(d1_select)
                if mol1: st.image(Draw.MolToImage(mol1, size=(400, 400)))
                
                # Hiển thị 3 dòng thông tin
                formula1 = get_mol_formula(d1_select)
                st.markdown(f"""
                <div style="{box_style} border-inline-start-color: #007bff;">
                    <small><b>Tên gọi:</b> {short_name1}</small><br>
                    <small><b>Công thức rút gọn:</b> {formula1}</small><br>
                    <small><b>Mã SMILES:</b></small><br>
                    <div style="{smiles_style}">{d1_select}</div>
                </div>
                """, unsafe_allow_html=True)

            # Thông tin thuốc 2
            with st_c2:
                mol2 = Chem.MolFromSmiles(d2_select)
                if mol2: st.image(Draw.MolToImage(mol2, size=(400, 400)))
                
                # Hiển thị 3 dòng thông tin
                formula2 = get_mol_formula(d2_select)
                st.markdown(f"""
                <div style="{box_style} border-inline-start-color: #28a745;">
                    <small><b>Tên gọi:</b> {short_name2}</small><br>
                    <small><b>Công thức rút gọn:</b> {formula2}</small><br>
                    <small><b>Mã SMILES:</b></small><br>
                    <div style="{smiles_style}">{d2_select}</div>
                </div>
                """, unsafe_allow_html=True)

            # Lấy ID và dữ liệu thực tế
            u_idx = drug_to_id.get(d1_select)
            v_idx = drug_to_id.get(d2_select)
            all_ints = df_full[(df_full['SMILES_1'] == d1_select) & (df_full['SMILES_2'] == d2_select)]

            # =========================================================
            # PHẦN A: KIỂM TRA CHỈ ĐỊNH CỤ THỂ (ƯU TIÊN HÀNG ĐẦU)
            # =========================================================
            if selected_side != "Tất cả":
                side_vn_target = side_vn_map.get(selected_side, selected_side)
                st.markdown(f"###  Kết quả phân tích cho: {side_vn_target}")
                
                # 1. Kiểm tra thực tế trong Dataset
                known_in_data = not all_ints[all_ints['Side_Name'] == selected_side].empty
                
                if known_in_data:
                    # Nếu đã có trong data thì BỎ QUA AI, báo xác nhận luôn
                    st.error(f"** Triệu chứng **{side_vn_target}** tồn tại với cặp thuốc này.")
                else:
                    # 2. Nếu chưa có trong data, AI mới thực hiện dự báo tập trung
                    r_idx = side_to_id.get(selected_side)
                    if model_ai and u_idx is not None and v_idx is not None and r_idx is not None:
                        try:
                            with torch.no_grad():
                                r_emb = model_ai.rel_emb(torch.tensor([r_idx], dtype=torch.long))
                                score = torch.sum(h_all[u_idx] * r_emb * h_all[v_idx])
                                prob_target = torch.sigmoid(score).item()
                        except Exception as e:
                            prob_target = 0.5
                            print(f"Model prediction error: {e}")
                        
                        if prob_target > 0.5:
                            st.warning(f"**Dự báo nguy cơ xuất hiện **{side_vn_target}** là **{prob_target*100:.1f}%**.")
                        else:
                            st.success(f"**ĐÁNH GIÁ:** Ít có khả năng xảy ra triệu chứng **{side_vn_target}** ({prob_target*100:.1f}%).")
            
            elif selected_side == "Tất cả":
                # ☑️ PHẦN A: HIỂN THỊ DỮ LIỆU THỰC TẾ (Nếu có)
                if not all_ints.empty:
                    st.markdown("### ✅ Triệu chứng ghi nhận chính thức")
                    st.info(f"💡 Cặp thuốc này có **{len(all_ints)}** tương tác đã được y văn ghi nhận.")
                    
                    # Tạo bảng chi tiết
                    # chỉ cần STT và tên triệu chứng (VN)
                    side_effects_found = sorted(all_ints['Side_Name'].unique(), key=lambda s: side_vn_map.get(s, s))
                    display_df = pd.DataFrame({
                        'STT': range(1, len(side_effects_found)+1),
                        'Tên triệu chứng': [side_vn_map.get(s, s) for s in side_effects_found]
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                else:
                    st.info("ℹ️ **Cặp thuốc này CHƯA CÓ trong dataset.** AI sẽ dự đoán các triệu chứng tiềm ẩn dựa trên mô hình.")

            # =========================================================
            # PHẦN B: AI DISCOVERY - TRÌNH BÀY BIỂU ĐỒ TRÒN CHI TIẾT (Chỉ khi "Tất cả" & cặp không có)
            # =========================================================
            st.markdown("---")
            st.markdown("##### 👁️ AI Discovery: Phân tích phân bố rủi ro tiềm ẩn")
            st.caption("Biểu đồ thể hiện tỷ lệ các mức xác suất dự báo trên tổng số 1,317 triệu chứng.")

            if model_ai and u_idx is not None and v_idx is not None:
                existing_sides = set(all_ints['Side_Name'].unique())
                if selected_side != "Tất cả": existing_sides.add(selected_side)

                try:
                    with torch.no_grad():
                        num_model = len(side_to_id)
                        probs_list = []
                        # r_emb will be computed inside loop
                        for r_idx in range(num_model):
                            r_emb = model_ai.rel_emb(torch.tensor([r_idx], dtype=torch.long))
                            score = torch.sum(h_all[u_idx] * r_emb * h_all[v_idx])
                            prob = torch.sigmoid(score).item()
                            probs_list.append(prob * 100)
                        all_probs_list = []
                        new_discoveries = []
                        side_names_list = list(side_to_id.keys())
                        for i in range(len(probs_list)):
                            side_en = side_names_list[i]
                            p_val = probs_list[i]
                            if side_en not in existing_sides:
                                all_probs_list.append(p_val)
                                new_discoveries.append((p_val, side_en))
                except Exception as e:
                    st.error(f"Lỗi khi tính toán AI Discovery: {e}")
                    all_probs_list = []
                    new_discoveries = []

                # --- KHU VỰC HIỂN THỊ BIỂU ĐỒ ---
                col_chart1, col_chart2 = st.columns([1.1, 0.9])

                with col_chart1:
                    st.write("**Top 10 rủi ro tiềm ẩn cao nhất**")
                    new_discoveries.sort(key=lambda x: x[0], reverse=True)
                    top_10 = new_discoveries[:10]
                    if top_10:
                        labels = [side_vn_map.get(x[1], x[1]) for x in top_10]
                        values = [x[0] for x in top_10]
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        bars = ax1.bar(labels, values, color='#3498db', alpha=0.8)
                        for bar in bars:
                            bar_height = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width()/2., bar_height + 1, f'{bar_height:.1f}%', ha='center', fontweight='bold', color='red')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig1)

                with col_chart2:
                    st.write("**Tỷ lệ phân bố theo mức xác suất**")
                    if all_probs_list:
                        df_p = pd.DataFrame(all_probs_list, columns=['Prob'])
                        # Làm tròn để nhóm các mức như 97.8% lại với nhau
                        df_p['Label'] = df_p['Prob'].round(1).astype(str) + "%"
                        counts = df_p['Label'].value_counts()
                        
                        # Chỉ lấy các mức xuất hiện nhiều nhất để biểu đồ tròn không bị nát
                        top_n = 6
                        if len(counts) > top_n:
                            counts_plot = counts[:top_n].copy()
                            counts_plot["Khác"] = counts[top_n:].sum()
                        else:
                            counts_plot = counts

                        fig2, ax2 = plt.subplots(figsize=(6, 6))
                        # Vẽ biểu đồ tròn chi tiết
                        wedges, texts, autotexts = ax2.pie(
                            counts_plot, 
                            labels=counts_plot.index, 
                            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                            startangle=140, 
                            colors=plt.cm.Pastel1.colors,
                            explode=[0.1 if "97.8" in str(idx) or "100" in str(idx) else 0 for idx in counts_plot.index]
                        )
                        plt.setp(autotexts, size=8, weight="bold")
                        st.pyplot(fig2)
                        
                        # Hiện số lượng cụ thể để giải thích cho biểu đồ
                        with st.expander("Xem số lượng triệu chứng chi tiết"):
                            for lbl, cnt in counts_plot.items():
                                st.write(f"- Mức **{lbl}**: {cnt} triệu chứng")
                    
# ==========================================
# 7. HIỂN THỊ DỮ LIỆU TỪ SIDEBAR ACTIONS
# ==========================================


# 2. Top 10 Tác dụng phụ - Phiên bản Thanh Ngang (Horizontal Bar)
if show_top_10:
    st.markdown("---")
    st.subheader("📊 Top 10 Tác dụng phụ xuất hiện nhiều nhất")
    
    # 1. Chuẩn bị dữ liệu (Lấy top 10 và đảo ngược để cái cao nhất nằm trên cùng)
    top_10_data = df_full['Side_Name'].value_counts().head(10).iloc[::-1]
    top_10_vn = [side_vn_map.get(x, x) for x in top_10_data.index]
    
    # 2. Vẽ bằng Matplotlib dạng thanh ngang
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Vẽ thanh ngang với màu xanh (giống trong hình bạn gửi)
    bars = ax.barh(top_10_vn, top_10_data.values, color='#4472C4', height=0.7)
    
    # 3. Thêm con số giá trị vào đầu mỗi thanh (giống hình mẫu)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(top_10_data.values)*0.01), # Vị trí x
                bar.get_y() + bar.get_height()/2,      # Vị trí y (giữa thanh)
                f'{int(width):,}',                     # Định dạng số có dấu phẩy
                va='center', fontweight='bold', color='#444')

    # 4. Tinh chỉnh giao diện để giống hình mẫu nhất
    ax.spines['top'].set_visible(False)    # Ẩn khung trên
    ax.spines['right'].set_visible(False)  # Ẩn khung phải
    ax.spines['bottom'].set_visible(False) # Ẩn khung dưới
    ax.xaxis.set_visible(False)            # Ẩn trục X luôn cho sạch

    # Tăng cỡ chữ cho nhãn bên trái (Tên triệu chứng)
    plt.yticks(fontsize=11, fontweight='bold', color='#333')
    
    # Hiển thị lên Streamlit
    st.pyplot(fig)

    
# 4. TOP 20 CẶP THUỐC NGUY HIỂM NHẤT (Nút mới thêm từ Sidebar)
if show_extreme:
    st.markdown("---")
    st.subheader("🚨 Top 20 cặp thuốc có nhiều triệu chứng tương tác nhất")
    
    with st.spinner("Đang trích xuất dữ liệu rủi ro cao..."):
        # Tính toán: Nhóm theo cặp và đếm số lượng triệu chứng
        top_20_pairs = (df_full.groupby(['SMILES_1', 'SMILES_2'])
                        .size()
                        .reset_index(name='Số lượng triệu chứng')
                        .nlargest(20, 'Số lượng triệu chứng'))
        
        # Ánh xạ tên thuốc
        top_20_pairs['Thuốc 1'] = top_20_pairs['SMILES_1'].apply(lambda x: drug_names.get(x, x))
        top_20_pairs['Thuốc 2'] = top_20_pairs['SMILES_2'].apply(lambda x: drug_names.get(x, x))
        
        # Hiển thị bảng
        st.dataframe(
            top_20_pairs[['Thuốc 1', 'Thuốc 2', 'Số lượng triệu chứng', 'SMILES_1', 'SMILES_2']],
            width="stretch",
            hide_index=True,
            column_config={
                "Số lượng triệu chứng": st.column_config.NumberColumn(
                    "Tổng số rủi ro",
                    format="%d 🚨",
                    help="Số lượng tác dụng phụ khác nhau được ghi nhận cho cặp này"
                ),
                "SMILES_1": None, # Ẩn cột SMILES cho bảng gọn đẹp
                "SMILES_2": None
            }
        )
        
        st.error("⚠️ **Lưu ý:** Đây là danh sách các cặp thuốc có mức độ tương tác phức tạp nhất trong hệ thống.")
        st.toast("Đã tải xong Top 20 cặp rủi ro cao!", icon="🚨")