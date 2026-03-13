# -*- coding: utf-8 -*-
import streamlit as st
import random


def render_sidebar(df_full, side_list, drug_to_id):
    with st.sidebar:
        st.header('BẢNG ĐIỀU KHIỂN')

        with st.expander('Thao tác nhanh', expanded=True):
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button('Random cặp đôi', use_container_width=True):
                    row = df_full.sample(n=1).iloc[0]
                    st.session_state['d1'] = row['SMILES_1']
                    st.session_state['d2'] = row['SMILES_2']
                    st.session_state['side'] = row['Side_Name']
                    st.session_state['d1_selectbox'] = row['SMILES_1']
                    st.session_state['d2_selectbox'] = row['SMILES_2']
                    st.session_state['side_selectbox'] = row['Side_Name']
                    st.rerun()
            with col_r2:
                if st.button('Random đơn chất', use_container_width=True):
                    picked = random.choice(list(drug_to_id.keys()))
                    st.session_state['d1'] = picked
                    st.session_state['d2'] = 'Trống'
                    st.session_state['side'] = 'Tất cả'
                    st.session_state['d1_selectbox'] = picked
                    st.session_state['d2_selectbox'] = 'Trống'
                    st.session_state['side_selectbox'] = 'Tất cả'
                    st.rerun()
            if st.button('Xóa chọn lựa', use_container_width=True):
                for key in ['d1','d2','side','d1_selectbox','d2_selectbox','side_selectbox']:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

        with st.expander('Bộ lọc dự đoán AI', expanded=True):
            prob_threshold = st.slider('Ngưỡng xác suất (%)', min_value=0, max_value=90, value=0, step=5)
            st.caption(f'Chỉ hiện triệu chứng có xác suất > {prob_threshold}%' if prob_threshold > 0 else 'Hiện tất cả triệu chứng')

        with st.expander('Thống kê & Báo cáo', expanded=True):
            show_top_10 = st.button('Top 10 Tác dụng phụ', use_container_width=True)
            show_extreme = st.button('Top 20 cặp rủi ro cao', use_container_width=True)
            show_analytics = st.button('Hiệu năng mô hình R-GCN', use_container_width=True)

        st.markdown('---')
        n_drugs = len(drug_to_id)
        n_sides = len(side_list) - 1
        n_pairs = len(df_full)
        st.info(f'Tổng số cặp: {n_pairs:,} | Số thuốc: {n_drugs:,} | Số TC: {n_sides:,} | Model: R-GCN')

    return show_top_10, show_extreme, prob_threshold, show_analytics
