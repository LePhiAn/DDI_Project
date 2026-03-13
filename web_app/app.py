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
    side_name_vn = side_vn_map.get(selected_side, "Tat ca rui ro")

    if d1 == "Trống" and d2 == "Trống":
        if selected_side == "Tất cả":
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Tong so cap", f"{len(df_full):,}")
            with k2:
                st.metric("So thuoc", f"{len(drug_to_id):,}")
            with k3:
                st.metric("Loai trieu chung", f"{len(side_list) - 1:,}")
            with k4:
                most_common = df_full["Side_Name"].value_counts().idxmax()
                st.metric("Pho bien nhat", side_vn_map.get(most_common, most_common)[:20])
            st.markdown("---")
            render_side_catalog_view(df_full, side_vn_map, drug_names)

        else:
            st.subheader(f"Tac dung phu: {side_name_vn}")
            full_pairs = df_full[df_full["Side_Name"] == selected_side]
            if not full_pairs.empty:
                st.markdown("### Cap ghi nhan trong dataset")
                st.info(f"Tim thay **{len(full_pairs):,}** cap thuoc voi tac dung phu nay.")
                pairs_display = full_pairs.copy()
                pairs_display["Thuoc 1"] = pairs_display["SMILES_1"].map(lambda x: drug_names.get(x, x))
                pairs_display["Thuoc 2"] = pairs_display["SMILES_2"].map(lambda x: drug_names.get(x, x))
                with st.expander("Xem bang toan bo cap"):
                    st.dataframe(pairs_display[["Thuoc 1", "Thuoc 2"]], use_container_width=True, hide_index=True)
            else:
                st.info("Chua co cap thuoc nao voi tac dung phu nay trong dataset.")

            st.markdown("---")
            st.markdown("### AI - Cac cap tiem an ngoai dataset")
            with st.spinner("Dang tinh toan AI..."):
                top_preds = predictor.get_top_unknown_pairs_for_side(selected_side, top_n=20)

            if top_preds:
                pred_data = [{
                    "prob": p,
                    "Xac suat (%)": f"{p:.1f}%",
                    "Thuoc 1": drug_names.get(d1s, d1s),
                    "Thuoc 2": drug_names.get(d2s, d2s)
                } for p, d1s, d2s in top_preds]
                pred_df = pd.DataFrame(pred_data)

                left_col, right_col = st.columns([1, 1])
                with left_col:
                    with st.expander("Bang top 20 du doan"):
                        st.dataframe(pred_df[["Xac suat (%)", "Thuoc 1", "Thuoc 2"]], use_container_width=True, hide_index=True)
                with right_col:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    names = pred_df.apply(lambda r: f"{r['Thuoc 1']} | {r['Thuoc 2']}", axis=1)
                    probs = pred_df["prob"].values
                    y_pos = list(range(len(names)))
                    ax.barh(y_pos, probs, color="tab:blue")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(names, fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel("Xac suat (%)")
                    ax.set_title("Top 20 cap du doan AI")
                    for i, v in enumerate(probs):
                        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Khong co cap tiem an duoc AI du doan.")

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
            st.error("Loi: Hai loai thuoc chon khong duoc trung nhau.")
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
    st.subheader("Top 10 Tac dung phu xuat hien nhieu nhat")
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
    st.subheader("Top 20 cap thuoc co nhieu trieu chung tuong tac nhat")
    with st.spinner("Dang trich xuat du lieu rui ro cao..."):
        top_20_pairs = (
            df_full.groupby(["SMILES_1", "SMILES_2"])
            .size()
            .reset_index(name="So luong trieu chung")
            .nlargest(20, "So luong trieu chung")
        )
        top_20_pairs["Thuoc 1"] = top_20_pairs["SMILES_1"].map(lambda x: drug_names.get(x, x))
        top_20_pairs["Thuoc 2"] = top_20_pairs["SMILES_2"].map(lambda x: drug_names.get(x, x))
        st.dataframe(
            top_20_pairs[["Thuoc 1", "Thuoc 2", "So luong trieu chung", "SMILES_1", "SMILES_2"]],
            use_container_width=True, hide_index=True,
            column_config={
                "So luong trieu chung": st.column_config.NumberColumn("Tong so rui ro", format="%d"),
                "SMILES_1": None,
                "SMILES_2": None
            }
        )
    st.error("Luu y: Day la danh sach cac cap thuoc co muc do tuong tac phuc tap nhat.")

if show_analytics:
    st.markdown("---")
    render_analytics_view()