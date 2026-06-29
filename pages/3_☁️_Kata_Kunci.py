"""
Halaman 3 — Word Cloud Kata Kunci
Visualisasi kata yang paling sering muncul per sentimen & platform.
"""
import io

import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

from shared import PLATFORM_LABELS, load_dataset

st.set_page_config(page_title="Kata Kunci — Sentiment Dashboard", page_icon="☁️", layout="wide")

st.title("☁️ Kata Kunci Ulasan")
st.caption("Kata-kata yang paling sering muncul berdasarkan sentimen dan platform.")

df = load_dataset()

if "score" not in df.columns or "text" not in df.columns:
    st.error("Kolom `score` atau `text` tidak ditemukan di dataset.")
    st.stop()

# Tambah kolom sentimen berdasarkan score
df = df.copy()
df["sentimen"] = df["score"].apply(lambda s: "Positif" if s >= 4 else ("Negatif" if s <= 2 else "Netral"))

# ── Filter Controls ──────────────────────────────────────────────────────────
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    sentiment_choice = st.radio(
        "Filter Sentimen:",
        options=["Semua", "Positif", "Negatif"],
        horizontal=True,
    )

with filter_col2:
    platform_options = ["Semua Platform"] + [
        PLATFORM_LABELS.get(p, p.capitalize()) for p in sorted(df["app_name"].unique())
    ]
    platform_choice = st.selectbox("Filter Platform:", options=platform_options)

# ── Filter data ──────────────────────────────────────────────────────────────
filtered = df.copy()

if sentiment_choice != "Semua":
    filtered = filtered[filtered["sentimen"] == sentiment_choice]

if platform_choice != "Semua Platform":
    platform_key = {v: k for k, v in PLATFORM_LABELS.items()}.get(platform_choice, platform_choice.lower())
    filtered = filtered[filtered["app_name"] == platform_key]

st.caption(f"Menampilkan **{len(filtered):,}** ulasan dari total {len(df):,}")

if filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# ── Generate word cloud ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Membuat word cloud...", max_entries=12)
def generate_wordcloud(texts: tuple, sentiment: str) -> bytes:
    """Generate word cloud dan kembalikan sebagai bytes PNG."""
    combined_text = " ".join(texts)

    # Stopwords Indonesia + default
    extra_stops = {
        "nya", "yang", "dan", "ke", "di", "aja", "bisa", "tidak", "ada",
        "mau", "jadi", "beli", "sudah", "terus", "saya", "kalau", "sama",
        "juga", "banyak", "hari", "banget", "sangat", "sekali", "selalu",
        "lah", "sih", "dong", "kan", "ga", "ya", "ini", "itu", "app",
        "tokopedia", "shopee", "lazada", "bukalapak", "aplikasi",
    }
    all_stops = STOPWORDS.union(extra_stops)

    color = "Greens" if sentiment == "Positif" else ("Reds" if sentiment == "Negatif" else "Blues")

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=color,
        stopwords=all_stops,
        max_words=120,
        collocations=True,
        prefer_horizontal=0.85,
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


texts_tuple = tuple(filtered["text"].astype(str).tolist())
img_bytes = generate_wordcloud(texts_tuple, sentiment_choice)

st.image(img_bytes, use_container_width=True)

# ── Top 20 kata terbanyak ─────────────────────────────────────────────────────
with st.expander("Lihat Top 20 Kata Terbanyak"):
    from collections import Counter
    import re

    stopwords_set = {
        "nya", "yang", "dan", "ke", "di", "aja", "bisa", "tidak", "ada",
        "mau", "jadi", "beli", "sudah", "terus", "saya", "kalau", "sama",
        "juga", "banyak", "hari", "banget", "sangat", "sekali", "selalu",
        "lah", "sih", "dong", "kan", "ga", "ya", "ini", "itu", "app",
        "tokopedia", "shopee", "lazada", "bukalapak", "aplikasi", "the",
        "and", "to", "a", "of", "in", "is",
    }

    all_words = re.findall(r'\b[a-z]{3,}\b', " ".join(texts_tuple).lower())
    filtered_words = [w for w in all_words if w not in stopwords_set]
    top_words = Counter(filtered_words).most_common(20)

    import pandas as pd
    top_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])
    st.dataframe(top_df, use_container_width=True, hide_index=True)
