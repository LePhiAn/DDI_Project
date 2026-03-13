# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

_GRAD = ["#1a5276","#1f618d","#2471a3","#2e86c1","#3498db","#5dade2","#7fb3d3","#a9cce3","#c5d9e8","#d6e9f5"]


def render_side_catalog_view(df_full, side_vn_map, drug_names):
    total_sides = df_full["Side_Name"].nunique()
    total_pairs = len(df_full)
    most_common_en = df_full["Side_Name"].value_counts().idxmax()
    most_common_vn = side_vn_map.get(most_common_en, most_common_en)

    k1, k2, k3 = st.columns(3)
    k1.metric("Tong loai trieu chung", f"{total_sides:,}")
    k2.metric("Tong so tuong tac", f"{total_pairs:,}")
    k3.metric("Pho bien nhat", most_common_vn[:25])

    st.markdown("---")
    st.markdown("### Top 10 Tac dung phu xuat hien nhieu nhat")
    top10 = df_full["Side_Name"].value_counts().head(10).iloc[::-1]
    top10_vn = [side_vn_map.get(x,x) for x in top10.index]

    fig, ax = plt.subplots(figsize=(12,5))
    colors = _GRAD[::-1][:len(top10)]
    bars = ax.barh(top10_vn, top10.values, color=colors, height=0.7, edgecolor="white", linewidth=0.8)
    avg = top10.values.mean()
    ax.axvline(x=avg, color="#e74c3c", linestyle="--", linewidth=1.3, alpha=0.8)
    ax.text(avg+max(top10.values)*0.005, len(top10)-0.3, f"TB: {avg:,.0f}", color="#e74c3c", fontsize=8, va="top")
    for bar, val in zip(bars, top10.values[::-1]):
        pct = val/top10.sum()*100
        ax.text(val+max(top10.values)*0.01, bar.get_y()+bar.get_height()/2,
                f"{int(val):,}  ({pct:.1f}%)", va="center", fontweight="bold", fontsize=9, color="#333")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False); ax.xaxis.set_visible(False)
    ax.set_xlim(0, max(top10.values)*1.3)
    plt.yticks(fontsize=10, fontweight="bold", color="#333")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("### Treemap - Phan bo Top 20 tac dung phu")
    st.caption("Kich thuoc o tuong ung voi so lan ghi nhan.")
    try:
        import squarify
        top20 = df_full["Side_Name"].value_counts().head(20)
        top20_vn = [side_vn_map.get(x,x)[:22] for x in top20.index]
        vals = top20.values.tolist()
        cmap = plt.cm.Blues
        norm_vals = [(v-min(vals))/(max(vals)-min(vals)+1) for v in vals]
        colors_tm = [cmap(0.3+0.6*nv) for nv in norm_vals]
        fig2, ax2 = plt.subplots(figsize=(12,5))
        squarify.plot(sizes=vals, label=top20_vn, color=colors_tm, alpha=0.85,
                      text_kwargs={"fontsize":8}, ax=ax2)
        ax2.axis("off"); ax2.set_title("Top 20 tac dung phu theo tan suat", fontsize=11, fontweight="bold")
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    except ImportError:
        st.info("Cai squarify de xem Treemap: pip install squarify")

    st.markdown("---")
    st.markdown("### Toan bo danh muc")
    search_query = st.text_input("Tim kiem tac dung phu (VN hoac EN):", placeholder="Nhap ten...")

    freq_counts = df_full["Side_Name"].value_counts()
    all_sides = sorted(freq_counts.index.tolist(), key=lambda x: side_vn_map.get(x,x).lower())
    if search_query:
        q = search_query.lower()
        all_sides = [s for s in all_sides if q in s.lower() or q in side_vn_map.get(s,"").lower()]

    st.markdown(f"**Tong so:** {len(all_sides)} tac dung phu")
    catalog_data = [{"STT":i,"Ten VN":side_vn_map.get(s,s),"Ten EN":s,"So lan ghi nhan":int(freq_counts.get(s,0))}
                    for i,s in enumerate(all_sides,1)]
    df_cat = pd.DataFrame(catalog_data)
    st.dataframe(df_cat, use_container_width=True, hide_index=True,
                 column_config={
                     "STT": st.column_config.NumberColumn("STT", width="small"),
                     "Ten VN": st.column_config.TextColumn("Ten tieng Viet", width="large"),
                     "Ten EN": st.column_config.TextColumn("Ten tieng Anh", width="large"),
                     "So lan ghi nhan": st.column_config.ProgressColumn(
                         "So lan ghi nhan", width="medium",
                         min_value=0, max_value=int(freq_counts.max()))
                 })
    st.download_button("Xuat danh muc CSV",
                       data=df_cat.to_csv(index=False, encoding="utf-8-sig"),
                       file_name="side_catalog.csv", mime="text/csv")