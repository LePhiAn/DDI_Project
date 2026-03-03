import streamlit as st
import random
import sys
import os

# For relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import truncate_drug_name


def render_input_section(drug_list):
    """Render drug selection section with search, validation, and random pair option."""
    
    st.markdown("### 💊 Chọn hai thuốc để phân tích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        drug1_display = st.selectbox(
            "**Thuốc thứ nhất:**",
            [truncate_drug_name(d, 45) for d in drug_list],
            key="drug1_select"
        )
        # Map back to full name
        drug1 = next(
            (d for d in drug_list if truncate_drug_name(d, 45) == drug1_display),
            drug1_display
        )
        if len(drug1) > 40:
            st.caption(f"📝 {drug1}")
    
    with col2:
        # Filter out drug1 from drug2 options
        drug2_options = [d for d in drug_list if d != drug1]
        drug2_display = st.selectbox(
            "**Thuốc thứ hai:**",
            [truncate_drug_name(d, 45) for d in drug2_options],
            key="drug2_select"
        )
        # Map back to full name
        drug2 = next(
            (d for d in drug2_options if truncate_drug_name(d, 45) == drug2_display),
            drug2_display
        )
        if len(drug2) > 40:
            st.caption(f"📝 {drug2}")
    
    button_col1, button_col2 = st.columns([1, 1])
    
    with button_col1:
        predict_button = st.button(
            "🔍 Phân tích tương tác",
            use_container_width=True,
            type="primary"
        )
    
    with button_col2:
        random_button = st.button(
            "🎲 Cặp kiểm thử ngẫu nhiên",
            use_container_width=True
        )
    
    # Handle random button
    if random_button and len(drug_list) >= 2:
        drug1, drug2 = random.sample(drug_list, 2)
        st.session_state.random_pair_label = True
        st.rerun()
    
    is_random = st.session_state.get("random_pair_label", False)
    if is_random:
        st.info("🎲 **Cặp kiểm thử ngoài tập huấn luyện**")
        st.session_state.random_pair_label = False

    return drug1, drug2, predict_button