"""
Halaman 5 — Coba Prediksi
Live inference: input teks bebas → prediksi sentimen + confidence score.
"""
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import preprocess_pipeline
from shared import get_model

st.set_page_config(page_title="Coba Prediksi — Sentiment Dashboard", page_icon="🔍", layout="centered")

st.title("🔍 Coba Prediksi Sentimen")
st.caption("Masukkan teks ulasan bebas dan lihat apakah model menilainya positif atau negatif.")

model, vectorizer, is_fallback = get_model()

if is_fallback:
    st.warning("⚠️ Model berjalan dalam mode fallback. Akurasi mungkin lebih rendah dari model asli.")


def predict_single(text: str):
    """Prediksi sentimen + confidence dari satu teks."""
    clean = preprocess_pipeline(text)
    if not clean:
        return None, None, None
    vec = vectorizer.transform([clean])
    label = int(model.predict(vec)[0])

    # Confidence via sigmoid dari decision_function
    score = model.decision_function(vec)[0]
    confidence = float(1 / (1 + np.exp(-abs(score)))) * 100

    label_str = "Positif" if label == 1 else "Negatif"
    return label_str, round(confidence, 1), clean


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Tulis ulasan di sini:",
    placeholder='Contoh: "Barangnya bagus, pengiriman cepat, packing rapi!"',
    height=120,
)

if st.button("🔎 Analisis Sentimen", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Tolong masukkan teks ulasan dulu ya.")
    else:
        label, confidence, clean_text = predict_single(user_input)

        if label is None:
            st.error("Teks tidak bisa diproses. Coba tulis ulasan yang lebih panjang.")
        else:
            # Tampilkan hasil
            if label == "Positif":
                st.success(f"### 👍 {label}  —  Confidence: {confidence}%")
            else:
                st.error(f"### 👎 {label}  —  Confidence: {confidence}%")

            with st.expander("Detail preprocessing"):
                st.code(f"Input asli  : {user_input}\nSetelah NLP : {clean_text}")

            # Simpan ke session history
            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].append({
                "Teks Asli": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                "Hasil": label,
                "Confidence": f"{confidence}%",
            })

# ── Riwayat prediksi sesi ini ────────────────────────────────────────────────
if "history" in st.session_state and st.session_state["history"]:
    st.divider()
    st.subheader("📋 Riwayat Prediksi Sesi Ini")

    history_df = pd.DataFrame(st.session_state["history"][::-1])  # terbaru di atas

    def style_hasil(val):
        if val == "Positif":
            return "color: #1b873f; font-weight: bold"
        return "color: #c0392b; font-weight: bold"

    st.dataframe(
        history_df.style.applymap(style_hasil, subset=["Hasil"]),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("🗑️ Hapus Riwayat"):
        st.session_state["history"] = []
        st.rerun()

st.divider()
st.caption(
    "🧠 Model: LinearSVC + TF-IDF (5000 vocab, unigram+bigram). "
    "Confidence dihitung dari `decision_function` dinormalisasi dengan fungsi sigmoid."
)
