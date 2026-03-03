# ============================================
# UI Helper Functions
# ============================================

import streamlit as st


def truncate_drug_name(name, max_length=40):
    """Truncate drug name if longer than max_length, add ... suffix."""
    if len(name) > max_length:
        return name[:max_length-3] + "..."
    return name


def format_stt(index):
    """Format number as 01, 02, 03..."""
    return f"{index+1:02d}"


def get_risk_color(risk_level):
    """Map risk level to hex color for badges."""
    colors = {
        "CRITICAL": "#DC3545",      # red
        "HIGH": "#FD7E14",          # orange
        "MODERATE": "#FFC107",      # amber/yellow
        "LOW": "#28A745",           # green
        "UNKNOWN": "#6C757D"        # gray
    }
    return colors.get(risk_level, colors["UNKNOWN"])


def render_risk_badge(risk_level):
    """Render a colored risk badge using HTML/CSS."""
    color = get_risk_color(risk_level)
    label_map = {
        "CRITICAL": "🔴 Rủi ro rất cao",
        "HIGH": "🟠 Rủi ro cao",
        "MODERATE": "🟡 Rủi ro trung bình",
        "LOW": "🟢 Rủi ro thấp",
        "UNKNOWN": "⚪ Không xác định"
    }
    label = label_map.get(risk_level, "Không xác định")
    html = f"""
    <span style='
        background-color: {color};
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    '>
        {label}
    </span>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_side_effect_item(index, side_effect_vi, side_effect_en, probability, confidence):
    """Render a single side effect item with Vietnamese name (large), English (small)."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{format_stt(index)}.** {side_effect_vi}")
        st.caption(f"_{side_effect_en}_")
    
    with col2:
        st.metric(
            label="Xác suất",
            value=f"{probability:.1%}",
            label_visibility="collapsed"
        )


def render_medical_warning():
    """Display medical safety disclaimer at the top of the page."""
    st.warning(
        "⚠️ **Cảnh báo y khoa**: Kết quả được tạo bởi mô hình AI dự đoán và "
        "**không thay thế tư vấn chuyên môn y tế**. Luôn tham khảo bác sĩ trước khi đưa ra quyết định "
        "liên quan đến sức khỏe. Mô hình này chỉ là công cụ hỗ trợ."
    )


def get_drug_display_names(drug_list, max_length=40):
    """Create a mapping of display names (truncated) to full names."""
    display_to_full = {}
    for drug in drug_list:
        truncated = truncate_drug_name(drug, max_length)
        display_to_full[truncated] = drug
    return display_to_full


def shorten_vietnamese_name(full_name, max_length=50):
    """Shorten Vietnamese text with ellipsis if too long."""
    if len(full_name) > max_length:
        return full_name[:max_length-3] + "..."
    return full_name
