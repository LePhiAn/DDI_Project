import streamlit as st
import sys
import os

# For relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import truncate_drug_name, format_stt, render_risk_badge


def render_pair_view(result, side_to_vn=None):
    """Render pair interaction results with Vietnamese labels and medical styling."""
    if "error" in result:
        st.error(f"❌ Lỗi: {result['error']}")
        return
    
    # ===== Header Card =====
    st.markdown("### 📊 Kết Quả Phân Tích Tương Tác")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        drug1_name = truncate_drug_name(result.get('drug_1', 'N/A'), 30)
        st.markdown(f"**Thuốc 1:** {drug1_name}")
        if len(result.get('drug_1', '')) > 30:
            st.caption(f"_{result.get('drug_1')}_")
    
    with col2:
        st.markdown("<div style='text-align: center; padding: 10px;'><strong style='font-size: 20px;'>+</strong></div>", unsafe_allow_html=True)
    
    with col3:
        drug2_name = truncate_drug_name(result.get('drug_2', 'N/A'), 30)
        st.markdown(f"**Thuốc 2:** {drug2_name}")
        if len(result.get('drug_2', '')) > 30:
            st.caption(f"_{result.get('drug_2')}_")
    
    st.markdown("---")
    
    # ===== Risk Assessment Card =====
    st.markdown("#### 🎯 Đánh Giá Rủi Ro")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.markdown("**Mức độ rủi ro:**")
        render_risk_badge(result.get("overall_risk_level", "UNKNOWN"))
    
    with risk_col2:
        confidence = result.get('confidence', result.get('overall_confidence', 0))
        st.metric("Độ tin cậy", f"{confidence*100:.1f}%")
    
    with risk_col3:
        probability = result.get('probability', 0)
        st.metric("Xác suất", f"{probability*100:.1f}%")
    
    st.markdown("---")
    
    # ===== Clinical Summary =====
    st.markdown("#### 📋 Giải Thích Lâm Sàng")
    explanation = result.get("clinical_summary", result.get("explanation", "Không có thông tin giải thích."))
    st.info(explanation)
    
    st.markdown("---")
    
    # ===== Top 5 Side Effects =====
    st.markdown("#### 🔴 Top 5 Tác Dụng Phụ Tiên Lượng")
    
    top_risks = result.get("top_10_risks", [])
    
    if not top_risks:
        st.info("Không có dự đoán tác dụng phụ.")
    else:
        for idx, item in enumerate(top_risks[:5]):
            st.markdown(f"""**{format_stt(idx)}. {item.get('side_effect', 'N/A')}**""")
            
            # English name in smaller text
            st.caption(item.get('side_effect', 'N/A'))
            
            # Metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                render_risk_badge(item.get('risk_level', 'UNKNOWN'))
            
            with metric_col2:
                st.metric("Xác suất", f"{item.get('probability', 0)*100:.1f}%", label_visibility="collapsed")
            
            with metric_col3:
                st.metric("Độ tin cậy", f"{item.get('confidence', 0)*100:.1f}%", label_visibility="collapsed")
            
            with metric_col4:
                pass  # Spacer
            
            st.divider()
    
    # ===== Out of Distribution Warning =====
    if result.get("is_out_of_distribution", False):
        st.warning("⚙️ **Cặp thuốc này ngoài tập dữ liệu huấn luyện** - Dự đoán mô hình có độ chắc chắn thấp hơn.")