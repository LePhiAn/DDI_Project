# -*- coding: utf-8 -*-
import streamlit as st


def render_input_section(drug_list, side_list, drug_names, side_vn_map):
    st.title("Phân Tích Tương Tác Thuốc")

    def fmt_drug(x):
        if x == "Trống": return "Trống"
        name = str(drug_names.get(x, "Unknown"))
        if name == "Unknown" or name.isdigit() or len(name) > 30:
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                mol = Chem.MolFromSmiles(x)
                return rdMolDescriptors.CalcMolFormula(mol) if mol else x
            except Exception:
                return x
        return name

    def fmt_side(x):
        if x == "Tất cả": return "Tất cả rủi ro"
        return side_vn_map.get(x, x)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            d1 = st.selectbox("Chọn thuốc thứ nhất:", drug_list,
                              format_func=fmt_drug, key="d1_selectbox")
        with col2:
            d2 = st.selectbox("Chọn thuốc thứ hai:", drug_list,
                              format_func=fmt_drug, key="d2_selectbox")
        selected_side = st.selectbox("Chỉ định tác dụng phụ:", side_list,
                                     format_func=fmt_side, key="side_selectbox")
        btn_analyze = st.button("PHÂN TÍCH", use_container_width=True, type="primary")

    return d1, d2, selected_side, btn_analyze
