"""
Halaman 1 — Ringkasan
Overview & KPI keseluruhan dataset + info model.
"""
import plotly.express as px
import streamlit as st

from shared import PLATFORM_COLORS, PLATFORM_LABELS, get_model, load_dataset

st.set_page_config(page_title="Ringkasan — Sentiment Dashboard", page_icon="📊", layout="wide")

st.title("📊 Ringkasan Dataset")
st.caption("Overview statistik seluruh 30.844 ulasan dari 4 platform e-commerce Indonesia.")

df = load_dataset()
model, vectorizer, is_fallback = get_model()

# ── Metric Cards ────────────────────────────────────────────────────────────
total_reviews = len(df)
date_min = df["review_datetime"].min().strftime("%b %Y") if "review_datetime" in df.columns else "N/A"
date_max = df["review_datetime"].max().strftime("%b %Y") if "review_datetime" in df.columns else "N/A"

# Hitung % positif dari kolom score (score >= 4 = positif), tanpa predict batch
if "score" in df.columns:
    positive_pct = round((df["score"] >= 4).mean() * 100, 1)
else:
    positive_pct = "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Ulasan", f"{total_reviews:,}")
col2.metric("% Positif (score ≥ 4)", f"{positive_pct}%")
col3.metric("Data Mulai", date_min)
col4.metric("Data Terakhir", date_max)

st.divider()

# ── Charts ──────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Distribusi Skor Ulasan (1–5)")
    if "score" in df.columns:
        score_counts = df["score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["Skor", "Jumlah"]
        score_counts["Skor"] = score_counts["Skor"].astype(str)
        fig_pie = px.pie(
            score_counts,
            names="Skor",
            values="Jumlah",
            color_discrete_sequence=px.colors.sequential.RdYlGn,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Kolom `score` tidak ditemukan di dataset.")

with chart_col2:
    st.subheader("Jumlah Ulasan per Platform")
    platform_counts = df["app_name"].value_counts().reset_index()
    platform_counts.columns = ["Platform", "Jumlah"]
    platform_counts["Platform"] = platform_counts["Platform"].map(
        lambda x: PLATFORM_LABELS.get(x, x.capitalize())
    )
    platform_counts["Warna"] = platform_counts["Platform"].map(
        lambda x: PLATFORM_COLORS.get(x.lower(), "#888888")
    )
    fig_bar = px.bar(
        platform_counts,
        x="Platform",
        y="Jumlah",
        color="Platform",
        color_discrete_map={
            PLATFORM_LABELS.get(k, k.capitalize()): v for k, v in PLATFORM_COLORS.items()
        },
        text_auto=True,
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Info Model ───────────────────────────────────────────────────────────────
st.subheader("ℹ️ Info Model")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Tipe Model", "LinearSVC")
info_col2.metric("Vocab TF-IDF", f"{len(vectorizer.vocabulary_):,} kata")
info_col3.metric("N-gram", "Unigram + Bigram")
info_col4.metric("Akurasi (test set)", "92.89%")

if is_fallback:
    st.warning("⚠️ Model berjalan dalam mode fallback (60 sampel). Letakkan `model_svm_best.pkl` di folder `models/` untuk hasil optimal.")
