# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

_HP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'training_history.json')

def _load():
    p = os.path.abspath(_HP)
    if not os.path.exists(p): return None
    with open(p, 'r', encoding='utf-8') as f: return json.load(f)

def render_analytics_view():
    st.subheader('Hiệu năng mô hình R-GCN')
    hist = _load()
    if hist is None:
        st.warning('Chưa có file training_history.json.'); return

    ep        = hist['epochs']
    loss      = hist['loss']
    val_loss  = hist.get('val_loss', [])
    train_auc = hist.get('train_auc', [])
    val_auc   = hist['val_auc']
    best_auc  = hist['best_val_auc']
    best_ep   = hist['best_epoch']

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Best Val AUC', f'{best_auc:.4f}', f'Epoch {best_ep}')
    k2.metric('Số Epoch', str(len(ep)))
    k3.metric('Số thuốc', str(hist.get('num_drugs', '-')))
    k4.metric('Số tác dụng phụ', str(hist.get('num_side_effects', '-')))
    st.markdown('---')

    tab_full, tab_skip = st.tabs(["📈 Toàn bộ (bao gồm Epoch 1–5)", "🔍 Chi tiết (bỏ Epoch 1–5)"])

    def _draw_loss_chart(ep_data, train_data, val_data, title):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        c_train = '#e74c3c'
        c_val   = '#e67e22'
        ax.plot(ep_data, train_data, color=c_train, linewidth=2, marker='o', markersize=3.5,
                label='Train Loss', alpha=0.9)
        ax.fill_between(ep_data, train_data, alpha=0.07, color=c_train)
        if val_data:
            ax.plot(ep_data, val_data, color=c_val, linewidth=2, marker='s', markersize=3.5,
                    linestyle='--', label='Test Loss', alpha=0.9)
            ax.fill_between(ep_data, val_data, alpha=0.05, color=c_val)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(ep_data[::2])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def _draw_auc_chart(ep_data, train_data, val_data, title):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        c_train = '#27ae60'
        c_val   = '#2980b9'
        if train_data:
            ax.plot(ep_data, train_data, color=c_train, linewidth=2, marker='o', markersize=3.5,
                    label='Train AUC', alpha=0.9)
            ax.fill_between(ep_data, train_data, alpha=0.07, color=c_train)
        ax.plot(ep_data, val_data, color=c_val, linewidth=2.5, marker='s', markersize=4,
                linestyle='--', label='Test AUC', alpha=0.9)
        ax.fill_between(ep_data, val_data, alpha=0.06, color=c_val)
        ax.axhline(y=best_auc, color=c_val, linestyle=':', linewidth=1.2, alpha=0.6)
        if best_ep in ep_data:
            ax.scatter([best_ep], [best_auc], color='gold', s=130, zorder=5,
                       edgecolors=c_val, linewidth=1.5, label=f'Best Ep {best_ep} ({best_auc:.4f})')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('AUC', fontsize=10)
        ax.set_ylim(min(val_data) * 0.98, max(val_data if not train_data else train_data) * 1.015)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(ep_data[::2])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_full:
        _draw_loss_chart(ep, loss, val_loss, 'Training Loss & Test Loss theo từng Epoch')
        st.markdown('')
        _draw_auc_chart(ep, train_auc, val_auc, 'Train AUC & Test AUC theo từng Epoch')

    with tab_skip:
        ep2        = ep[5:]
        loss2      = loss[5:]
        val_loss2  = val_loss[5:] if val_loss else []
        train_auc2 = train_auc[5:] if train_auc else []
        val_auc2   = val_auc[5:]
        _draw_loss_chart(ep2, loss2, val_loss2, 'Training Loss & Test Loss (bỏ Epoch 1–5)')
        st.markdown('')
        _draw_auc_chart(ep2, train_auc2, val_auc2, 'Train AUC & Test AUC (bỏ Epoch 1–5)')
        st.caption(f'⚠️ Epoch 1–5 bị ẩn (Loss từ {loss[0]:,.0f} → {loss[4]:,.1f}) để thấy rõ xu hướng hội tụ.')

    st.markdown('---')
    st.markdown('### Chi tiết từng Epoch')
    dl = ['-'] + [f'{val_auc[i] - val_auc[i-1]:+.4f}' for i in range(1, len(val_auc))]
    rows = {
        'Epoch': ep,
        'Train Loss': [f'{x:.4f}' for x in loss],
        'Test Loss':  [f'{x:.4f}' for x in val_loss] if val_loss else ['-'] * len(ep),
        'Train AUC':  [f'{x:.4f}' for x in train_auc] if train_auc else ['-'] * len(ep),
        'Test AUC':   [f'{x:.4f}' for x in val_auc],
        'Δ Test AUC': dl,
    }
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
