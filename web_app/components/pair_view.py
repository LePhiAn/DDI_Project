# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

_BOX = "text-align:start;background-color:#f8f9fa;padding:15px;border-radius:10px;border-inline-start:5px solid;"
_SML = "word-break:break-all;color:#e83e8c;font-size:11px;background-color:#f1f1f1;padding:2px 5px;border-radius:3px;"


def _formula(s):
    try:
        m = Chem.MolFromSmiles(s)
        return rdMolDescriptors.CalcMolFormula(m) if m else s
    except: return s


def _risk_color(p):
    if p >= 70: return "#e74c3c"
    if p >= 40: return "#e67e22"
    return "#27ae60"


def _gauge(val, ax):
    theta = np.linspace(np.pi, 0, 300)
    ax.set_aspect("equal")
    for i, (t1, t2) in enumerate(zip(theta[:-1], theta[1:])):
        ax.fill_between([np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)], 0,
                        color=plt.cm.RdYlGn_r(i/len(theta)), alpha=0.85)
    ang = np.pi - (val/100)*np.pi
    ax.annotate("", xy=(np.cos(ang)*0.7, np.sin(ang)*0.7), xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.15,1.1)
    ax.text(0,-0.08,f"{val:.0f}%",ha="center",va="center",
            fontsize=14,fontweight="bold",color=_risk_color(val))
    ax.axis("off")


def render_pair_view(d1_smiles, d2_smiles, selected_side, df_full,
                     drug_names, side_vn_map, predictor, prob_threshold=0):
    short1 = drug_names.get(d1_smiles, _formula(d1_smiles))
    short2 = drug_names.get(d2_smiles, _formula(d2_smiles))
    st.subheader(f"Doi chieu: {short1} & {short2}")

    c1, c2 = st.columns(2)
    for col, sm, name, color in [(c1,d1_smiles,short1,"#007bff"),(c2,d2_smiles,short2,"#28a745")]:
        with col:
            mol = Chem.MolFromSmiles(sm)
            if mol: st.image(Draw.MolToImage(mol, size=(400,400)))
            st.markdown(f'<div style="{_BOX} border-inline-start-color:{color};"><small><b>Ten:</b> {name}</small><br><small><b>CT:</b> {_formula(sm)}</small><br><div style="{_SML}">{sm}</div></div>', unsafe_allow_html=True)

    all_ints = df_full[(df_full["SMILES_1"]==d1_smiles)&(df_full["SMILES_2"]==d2_smiles)]

    if selected_side != "Tat ca":
        side_vn = side_vn_map.get(selected_side, selected_side)
        st.markdown(f"### Ket qua: {side_vn}")
        known = not all_ints[all_ints["Side_Name"]==selected_side].empty
        if known:
            st.error(f"CO TRONG DATASET: Trieu chung {side_vn} da duoc ghi nhan chinh thuc.")
        else:
            prob = predictor.get_prob(d1_smiles, d2_smiles, selected_side)
            if prob is not None:
                pp = prob*100
                c = _risk_color(pp)
                st.markdown(f'<div style="border:2px solid {c};border-radius:8px;padding:12px;"><b style="color:{c};">CHUA CO TRONG DATASET</b><br>Xac suat AI: <b style="color:{c};font-size:16px;">{pp:.1f}%</b></div>', unsafe_allow_html=True)
                fig_g,ax_g = plt.subplots(figsize=(4,2.2))
                _gauge(pp,ax_g); ax_g.set_title("Risk Score",fontsize=9)
                st.pyplot(fig_g); plt.close(fig_g)
            else:
                st.warning("Khong the chay AI du doan.")
    else:
        if not all_ints.empty:
            st.markdown("### Trieu chung ghi nhan chinh thuc")
            st.info(f"Cap thuoc nay co {len(all_ints)} tuong tac da ghi nhan.")
            sides = sorted(all_ints["Side_Name"].unique(), key=lambda s: side_vn_map.get(s,s))
            st.dataframe(pd.DataFrame({"STT":range(1,len(sides)+1),"Trieu chung":[side_vn_map.get(s,s) for s in sides]}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Cap thuoc nay CHUA CO trong dataset. AI se du doan.")

    st.markdown("---")
    st.markdown("##### AI Discovery: Phan tich rui ro tiem an")
    st.caption("Mau sac: Do >= 70% | Cam 40-70% | Xanh < 40%")

    existing = set(all_ints["Side_Name"].unique())
    if selected_side != "Tat ca": existing.add(selected_side)

    try:
        all_preds = predictor.get_all_side_probs(d1_smiles, d2_smiles, exclude_sides=existing)
        discoveries = [(p,s) for p,s in all_preds if p > prob_threshold]
        if not discoveries:
            st.info(f"Khong co trieu chung vuot nguong {prob_threshold:.0f}%.")
        else:
            top10 = discoveries[:10]
            labels = [side_vn_map.get(s,s) for _,s in top10]
            values = [p for p,_ in top10]
            bar_colors = [_risk_color(v) for v in values]
            fig, ax = plt.subplots(figsize=(9, max(3, len(top10)*0.65)))
            bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, values[::-1]):
                ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color=_risk_color(val))
            ax.axvline(x=50, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(50.5, len(top10)-0.5, "50%", fontsize=8, color="#7f8c8d", va="top")
            ax.set_xlim(0, max(max(values)*1.25, 60))
            ax.set_xlabel("Xac suat du bao (%)", fontsize=9)
            ax.set_title(f"Top {len(top10)} rui ro tiem an (nguong >{prob_threshold:.0f}%)", fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)
            patches = [mpatches.Patch(color="#e74c3c",label="Nguy hiem cao (>=70%)"),
                       mpatches.Patch(color="#e67e22",label="Trung binh (40-70%)"),
                       mpatches.Patch(color="#27ae60",label="Thap (<40%)")]
            ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            if values:
                overall = np.average(values, weights=range(len(values),0,-1))
                cg, cs = st.columns([1,2])
                with cg:
                    st.markdown("**Risk Score tong the**")
                    fig_g,ax_g = plt.subplots(figsize=(3.5,2))
                    _gauge(overall,ax_g); st.pyplot(fig_g); plt.close(fig_g)
                with cs:
                    high=sum(1 for v in values if v>=70)
                    mid=sum(1 for v in values if 40<=v<70)
                    low=sum(1 for v in values if v<40)
                    st.markdown(f"**Phan bo muc nguy hiem (Top {len(top10)}):**\n- Do (>=70%): **{high}**\n- Cam (40-70%): **{mid}**\n- Xanh (<40%): **{low}**\n\nTong: **{len(discoveries)}** trieu chung vuot nguong {prob_threshold:.0f}%")

            if discoveries:
                df_exp = pd.DataFrame([(side_vn_map.get(s,s),s,f"{p:.2f}%","Cao" if p>=70 else ("Tb" if p>=40 else "Thap")) for p,s in discoveries],
                                      columns=["Trieu chung (VN)","Ten goc","Xac suat AI","Muc nguy hiem"])
                st.download_button("Xuat ket qua CSV",
                                   data=df_exp.to_csv(index=False, encoding="utf-8-sig"),
                                   file_name=f"ai_{short1}_vs_{short2}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Loi khi tinh toan AI Discovery: {e}")