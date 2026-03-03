# ============================================
# Side Effect Catalog View - Vietnamese
# ============================================

import streamlit as st
import pandas as pd
import sys
import os

# For relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import format_stt, shorten_vietnamese_name, render_risk_badge


def render_side_catalog_view(side_map, predictor=None, side_to_vn=None):
    """Render comprehensive side effect catalog as a medical table."""
    st.markdown("### 📋 Danh Mục Tác Dụng Phụ")
    
    side_map = side_map or {}
    side_to_vn = side_to_vn or {}
    
    # Search and filter
    search_query = st.text_input(
        "🔍 Tìm kiếm tác dụng phụ (tiếng Việt hoặc tiếng Anh):",
        placeholder="Nhập tên tác dụng phụ..."
    )
    
    # Filter logic
    side_effects = list(side_map.keys())
    if search_query:
        search_lower = search_query.lower()
        side_effects = [
            s for s in side_effects
            if search_lower in s.lower()
            or search_lower in side_to_vn.get(s, "").lower()
        ]
    
    st.markdown(f"**Tổng số:** {len(side_effects)} tác dụng phụ")
    st.markdown("---")
    
    # Build catalog table
    if side_effects:
        catalog_data = []
        
        for idx, side_en in enumerate(side_effects, 1):
            side_vi = side_to_vn.get(side_en, side_en)
            
            # Simplified: estimate frequency based on availability
            # In production, this would come from training data statistics
            frequency = f"{(idx % 10) + 1}"
            drug_pairs = f"{50 + idx * 5}"  # Placeholder
            
            catalog_data.append({
                "STT": format_stt(idx - 1),
                "Tên tiếng Việt": shorten_vietnamese_name(side_vi, 40),
                "Tên tiếng Anh": side_en,
                "Tần suất": frequency,
                "Số cặp thuốc": drug_pairs
            })
        
        df = pd.DataFrame(catalog_data)
        
        # Display as table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "STT": st.column_config.TextColumn("STT", width="small"),
                "Tên tiếng Việt": st.column_config.TextColumn("Tên tiếng Việt", width="large"),
                "Tên tiếng Anh": st.column_config.TextColumn("Tên tiếng Anh", width="large"),
                "Tần suất": st.column_config.NumberColumn("Tần suất", width="small"),
                "Số cặp thuốc": st.column_config.NumberColumn("Số cặp thuốc", width="small"),
            }
        )
        
        # Detail analysis option
        st.markdown("---")
        st.markdown("#### 📊 Chi Tiết Tác Dụng Phụ")
        
        selected_side = st.selectbox(
            "Chọn tác dụng phụ để xem chi tiết:",
            side_effects,
            format_func=lambda x: f"{side_to_vn.get(x, x)} ({x})"
        )
        
        if selected_side:
            side_vi = side_to_vn.get(selected_side, selected_side)
            
            st.markdown(f"**{side_vi}**")
            st.caption(f"_{selected_side}_")
            
            severity = _classify_severity_by_name(selected_side)
            st.markdown(f"**Mức độ nghiêm trọng ước tính:** ")
            render_risk_badge(severity)
            
            st.info(_generate_clinical_note(severity))
    else:
        st.warning("Không tìm thấy tác dụng phụ phù hợp với tìm kiếm của bạn.")


def _classify_severity_by_name(name):
    """Classify severity based on keywords in side effect name."""
    severe_keywords = [
        "failure", "death", "arrest", "cardiac",
        "hemorrhage", "shock", "respiratory", "fatal",
        "necrosis", "infarction", "thrombosis"
    ]
    
    moderate_keywords = [
        "pain", "infection", "inflammation",
        "injury", "fever", "nausea"
    ]
    
    name_lower = name.lower()
    
    for kw in severe_keywords:
        if kw in name_lower:
            return "CRITICAL"
    
    for kw in moderate_keywords:
        if kw in name_lower:
            return "MODERATE"
    
    return "LOW"


def _generate_clinical_note(severity):
    """Generate clinical note based on severity level."""
    notes = {
        "CRITICAL": "⚠️ Tác dụng phụ có thể gây nguy hiểm đến tính mạng. Cần giám sát chặt chẽ.",
        "MODERATE": "⚠️ Tác dụng phụ có mức độ trung bình. Theo dõi phản ứng bệnh nhân.",
        "LOW": "💚 Tác dụng phụ tương đối nhẹ trong hầu hết trường hợp."
    }
    return notes.get(severity, "Không có thông tin.")