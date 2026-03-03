import streamlit as st


def render_sidebar():
    st.sidebar.markdown("# 💊 Phân Tích Tương Tác Thuốc")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "**Chọn chức năng:**",
        [
            "🔄 Phân tích cặp",
            "💉 Thông tin thuốc",
            "📋 Danh mục tác dụng phụ",
            "📊 Phân tích tổng quát"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>⚠️ Mô hình AI - không thay thế tư vấn bác sĩ</small>",
        unsafe_allow_html=True
    )

    return menu