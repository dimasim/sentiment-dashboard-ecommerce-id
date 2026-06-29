"""
Halaman 2 — Perbandingan Platform
Positive rate & breakdown skor per platform.
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shared import PLATFORM_COLORS, PLATFORM_LABELS, load_dataset

st.set_page_config(page_title="Perbandingan — Sentiment Dashboard", page_icon="🏪", layout="wide")

st.title("🏪 Perbandingan Platform")
st.caption("Bandingkan tingkat kepuasan pengguna antar platform berdasarkan skor ulasan.")

df = load_dataset()

if "score" not in df.columns:
    st.error("Kolom `score` tidak ditemukan di dataset.")
    st.stop()

# ── Hitung statistik per platform ───────────────────────────────────────────
platforms = sorted(df["app_name"].unique())
rows = []
for p in platforms:
    sub = df[df["app_name"] == p]
    total = len(sub)
    positif = (sub["score"] >= 4).sum()
    negatif = (sub["score"] <= 2).sum()
    netral = total - positif - negatif
    rows.append({
        "platform_key": p,
        "Platform": PLATFORM_LABELS.get(p, p.capitalize()),
        "Total": total,
        "Positif": int(positif),
        "Negatif": int(negatif),
        "Netral": int(netral),
        "% Positif": round(positif / total * 100, 2),
        "Warna": PLATFORM_COLORS.get(p, "#888888"),
    })

import pandas as pd
stats_df = pd.DataFrame(rows).sort_values("% Positif", ascending=True)

# ── Chart 1: Positive rate horizontal bar ───────────────────────────────────
st.subheader("Tingkat Kepuasan per Platform")

fig_pos = go.Figure()
for _, row in stats_df.iterrows():
    fig_pos.add_trace(go.Bar(
        x=[row["% Positif"]],
        y=[row["Platform"]],
        orientation="h",
        name=row["Platform"],
        marker_color=row["Warna"],
        text=f"{row['% Positif']}%",
        textposition="outside",
    ))

# Threshold line 50%
fig_pos.add_vline(x=50, line_dash="dash", line_color="gray",
                  annotation_text="50% threshold", annotation_position="top right")
fig_pos.update_layout(
    showlegend=False,
    xaxis=dict(range=[0, 100], title="% Ulasan Positif (skor ≥ 4)"),
    yaxis_title="",
    height=300,
    margin=dict(t=20, b=20, l=20, r=60),
)
st.plotly_chart(fig_pos, use_container_width=True)

st.divider()

# ── Chart 2: Stacked bar breakdown skor 1-5 ─────────────────────────────────
st.subheader("Distribusi Skor per Platform")

score_detail = []
for p in platforms:
    sub = df[df["app_name"] == p]
    total = len(sub)
    label = PLATFORM_LABELS.get(p, p.capitalize())
    for s in [1, 2, 3, 4, 5]:
        count = (sub["score"] == s).sum()
        score_detail.append({
            "Platform": label,
            "Skor": f"⭐ {s}",
            "Jumlah": int(count),
            "Persen": round(count / total * 100, 1),
        })

detail_df = pd.DataFrame(score_detail)
fig_stack = px.bar(
    detail_df,
    x="Platform",
    y="Persen",
    color="Skor",
    barmode="stack",
    text="Persen",
    color_discrete_sequence=["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1b873f"],
)
fig_stack.update_traces(texttemplate="%{text}%", textposition="inside")
fig_stack.update_layout(
    yaxis_title="Persentase (%)",
    height=380,
    margin=dict(t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_stack, use_container_width=True)

st.divider()

# ── Tabel Ringkasan ──────────────────────────────────────────────────────────
st.subheader("Tabel Ringkasan")
display_df = stats_df[["Platform", "Total", "Positif", "Negatif", "Netral", "% Positif"]].copy()
display_df = display_df.sort_values("% Positif", ascending=False).reset_index(drop=True)

def color_pct(val):
    if val >= 50:
        return "color: #1b873f; font-weight: bold"
    return "color: #c0392b; font-weight: bold"

st.dataframe(
    display_df.style.applymap(color_pct, subset=["% Positif"]).format({"% Positif": "{:.2f}%", "Total": "{:,}", "Positif": "{:,}", "Negatif": "{:,}", "Netral": "{:,}"}),
    use_container_width=True,
    hide_index=True,
)
