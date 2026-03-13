# -*- coding: utf-8 -*-
import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

WEB_DIR = os.path.dirname(__file__)
if WEB_DIR not in sys.path:
    sys.path.insert(0, WEB_DIR)

from services.model_loader import ModelLoader
from services.mapping_service import MappingService
from services.predictor_service import PredictorService

from components.sidebar import render_sidebar
from components.input_section import render_input_section
from components.single_drug_view import render_single_drug_view
from components.pair_view import render_pair_view
from components.side_catalog_view import render_side_catalog_view
from components.analytics_view import render_analytics_view

st.set_page_config(page_title="AI Pharma - Drug Interaction Insights", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { cursor: default; user-select: none; }
    div[data-testid="stSelectbox"] input { cursor: text !important; user-select: auto !important; }
    h1, h2, h3, h4, h5, h6, p, span, div { -webkit-user-select: none; user-select: none; }
    input, textarea { cursor: text !important; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    mapping = MappingService()
    drug_to_id, side_to_id = mapping.get_maps()

    model_path = os.path.join(ROOT_DIR, "models/r_gcn_full_model.pth")
    loader = ModelLoader(
        model_path=model_path,
        num_nodes=mapping.num_nodes,
        num_relations=mapping.num_relations,
        hidden_channels=64,
        embedding_dim=16,
        device="cpu"
    )
    model = loader.get_model()

    predictor = PredictorService(
        model=model,
        drug_to_id=drug_to_id,
        side_to_id=side_to_id,
        df_full=mapping.df,
        device="cpu",
        side_to_vn=mapping.side_to_vn
    )

    return mapping, predictor


mapping, predictor = load_all()

df_full      = mapping.df
drug_list    = mapping.get_drug_list()
side_list    = mapping.get_side_options()
drug_names   = mapping.drug_to_name
side_vn_map  = mapping.side_to_vn
drug_to_id   = mapping.drug_to_id

if "d1"   not in st.session_state: st.session_state.d1   = "Trống"
if "d2"   not in st.session_state: st.session_state.d2   = "Trống"
if "side" not in st.session_state: st.session_state.side = "Tất cả"

show_top_10, show_extreme, prob_threshold, show_analytics = render_sidebar(df_full, side_list, drug_to_id)

d1, d2, selected_side, btn_analyze = render_input_section(
    drug_list, side_list, drug_names, side_vn_map
)

if btn_analyze:
    st.markdown("---")
    side_name_vn = side_vn_map.get(selected_side, "Tất cả rủi ro")

    if d1 == "Trống" and d2 == "Trống":
        if selected_side == "Tất cả":
            render_side_catalog_view(df_full, side_vn_map, drug_names)

        else:
            st.subheader(f"Tác dụng phụ: {side_name_vn}")
            full_pairs = df_full[df_full["Side_Name"] == selected_side]
            if not full_pairs.empty:
                st.markdown("### Cặp ghi nhận trong dataset")
                st.info(f"Tìm thấy **{len(full_pairs):,}** cặp thuốc với tác dụng phụ này.")
                pairs_display = full_pairs.copy()
                pairs_display["Thuoc 1"] = pairs_display["SMILES_1"].map(lambda x: drug_names.get(x, x))
                pairs_display["Thuoc 2"] = pairs_display["SMILES_2"].map(lambda x: drug_names.get(x, x))
                with st.expander("Xem bảng toàn bộ cặp"):
                    st.dataframe(pairs_display[["Thuoc 1", "Thuoc 2"]], use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có cặp thuốc nào với tác dụng phụ này trong dataset.")

            st.markdown("---")
            st.markdown("### AI - Các cặp tiềm ẩn ngoài dataset")
            with st.spinner("Đang tính toán AI..."):
                top_preds = predictor.get_top_unknown_pairs_for_side(selected_side, top_n=20)

            if top_preds:
                pred_data = [{
                    "prob": p,
                    "Xác suất (%)": f"{p:.1f}%",
                    "Thuoc 1": drug_names.get(d1s, d1s),
                    "Thuoc 2": drug_names.get(d2s, d2s)
                } for p, d1s, d2s in top_preds]
                pred_df = pd.DataFrame(pred_data)

                left_col, right_col = st.columns([1, 1])
                with left_col:
                    with st.expander("Bảng top 20 dự đoán"):
                        st.dataframe(pred_df[["Xác suất (%)", "Thuoc 1", "Thuoc 2"]], use_container_width=True, hide_index=True)
                with right_col:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    names = pred_df.apply(lambda r: f"{r['Thuoc 1']} | {r['Thuoc 2']}", axis=1)
                    probs = pred_df["prob"].values
                    y_pos = list(range(len(names)))
                    ax.barh(y_pos, probs, color="tab:blue")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(names, fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel("Xác suất (%)")
                    ax.set_title("Top 20 cặp dự đoán AI")
                    for i, v in enumerate(probs):
                        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Không có cặp tiềm ẩn được AI dự đoán.")

    elif d1 == "Trống" or d2 == "Trống":
        active_drug = d1 if d1 != "Trống" else d2
        render_single_drug_view(
            active_drug=active_drug,
            selected_side=selected_side,
            df_full=df_full,
            drug_names=drug_names,
            side_vn_map=side_vn_map,
            predictor=predictor
        )

    else:
        if d1 == d2:
            st.error("Lỗi: Hai loại thuốc chọn không được trùng nhau.")
        else:
            render_pair_view(
                d1_smiles=d1, d2_smiles=d2,
                selected_side=selected_side,
                df_full=df_full,
                drug_names=drug_names,
                side_vn_map=side_vn_map,
                predictor=predictor,
                prob_threshold=prob_threshold
            )


if show_top_10:
    st.markdown("---")
    st.subheader("Top 10 Tác dụng phụ xuất hiện nhiều nhất")
    top_10_data = df_full["Side_Name"].value_counts().head(10).iloc[::-1]
    top_10_vn = [side_vn_map.get(x, x) for x in top_10_data.index]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_10_vn, top_10_data.values, color="#4472C4", height=0.7)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(top_10_data.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontweight="bold", color="#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_visible(False)
    plt.yticks(fontsize=11, fontweight="bold", color="#333")
    st.pyplot(fig)
    plt.close(fig)

if show_extreme:
    st.markdown("---")
    st.subheader("Top 20 cặp thuốc có nhiều triệu chứng tương tác nhất")
    with st.spinner("Đang trích xuất dữ liệu rủi ro cao..."):
        top_20_pairs = (
            df_full.groupby(["SMILES_1", "SMILES_2"])
            .size()
            .reset_index(name="Số lượng triệu chứng")
            .nlargest(20, "Số lượng triệu chứng")
        )
        top_20_pairs["Thuốc 1"] = top_20_pairs["SMILES_1"].map(lambda x: drug_names.get(x, x))
        top_20_pairs["Thuốc 2"] = top_20_pairs["SMILES_2"].map(lambda x: drug_names.get(x, x))
        st.dataframe(
            top_20_pairs[["Thuốc 1", "Thuốc 2", "Số lượng triệu chứng", "SMILES_1", "SMILES_2"]],
            use_container_width=True, hide_index=True,
            column_config={
                "Số lượng triệu chứng": st.column_config.NumberColumn("Tổng số rủi ro", format="%d"),
                "SMILES_1": None,
                "SMILES_2": None
            }
        )
    st.error("Lưu ý: Đây là danh sách các cặp thuốc có mức độ tương tác phức tạp nhất.")

if show_analytics:
    st.markdown("---")
    render_analytics_view()