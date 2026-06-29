"""
Asisten Belanja: Cek Sentimen Ulasan Olshop
============================================
Aplikasi untuk membantu pembeli (terutama ibu rumah tangga) memutuskan
apakah sebuah produk/toko aman dibeli, berdasarkan ringkasan otomatis
dari kumpulan ulasan pembeli sebelumnya.

Jalankan dengan: streamlit run app.py
"""
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import streamlit as st

from model_utils import load_or_train_model
from preprocessing import preprocess_pipeline, split_reviews

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Cek Sentimen Ulasan Olshop",
    page_icon="🛍️",
    layout="centered",
)

# Kata-kata kunci untuk diidentifikasi sebagai "topik keluhan" pada ulasan negatif.
# Dipetakan ke kategori yang mudah dipahami orang awam.
COMPLAINT_KEYWORDS = {
    "Pengiriman lambat": ["lama", "lambat", "telat", "lelet"],
    "Barang rusak/cacat": ["rusak", "cacat", "sobek", "penyok", "pecah"],
    "Tidak sesuai pesanan": ["sesuai", "beda", "salah", "ga sesuai", "tidak sesuai"],
    "Kualitas kurang baik": ["jelek", "buruk", "murahan", "zonk"],
    "Pelayanan/CS kurang responsif": ["respon", "cs", "komplain", "balas"],
    "Kemasan/packing kurang baik": ["packing", "kemasan", "bungkus"],
}


# Kata-kata yang diexclude karena tidak informatif (Custom Stopwords)
CUSTOM_STOPWORDS = [
    # Nama platform itu sendiri
    'tokopedia', 'tokped', 'shopee', 'shope', 'lazada', 'bukalapak', 'lapak', 'buka',
    # Sufiks/partikel yang lolos stemming
    'nya', 'app', 'di', 'yang', 'dan', 'ke', 'di',
    # Kata generik tanpa nilai insight
    'aja', 'bisa', 'tidak', 'ada', 'mau', 'jadi', 'beli',
    'sudah', 'terus', 'lama', 'saya', 'kalau', 'sama',
    'sekarang', 'dulu', 'juga', 'banyak', 'hari', 'kali',
    'banget', 'sangat', 'sekali', 'selalu', 'malah',
    'padahal', 'jelas', 'buat', 'pakai', 'toko', 'apa',
    'lebih', 'semua', 'lah', 'sih', 'dong', 'kan',
]


@st.cache_data(show_spinner="Memuat data ulasan...")
def load_dataset():
    """Memuat dataset ulasan dari file CSV."""
    try:
        df = pd.read_csv("data/dataset_gabungan.csv", low_memory=False)
        # Pastikan kolom yang diperlukan ada
        if 'app_name' not in df.columns or 'text' not in df.columns:
            return pd.DataFrame(columns=['app_name', 'text'])
        df['app_name'] = df['app_name'].str.lower().str.strip()
        return df
    except Exception:
        return pd.DataFrame(columns=['app_name', 'text'])


@st.cache_resource(show_spinner=False)
def get_model():
    return load_or_train_model()


@st.cache_data(show_spinner=False)
def get_predictions_cached(app_name: str) -> list[tuple]:
    """
    Predict sentimen seluruh ulasan untuk satu platform — di-cache per platform.
    Jauh lebih efisien daripada predict ulang setiap interaksi.
    """
    model, vectorizer, _ = get_model()
    df = load_dataset()
    df_sel = df[df['app_name'] == app_name]
    reviews = df_sel['text'].astype(str).tolist()
    return predict_sentiments(reviews, model, vectorizer)


def predict_sentiments(reviews: list[str], model, vectorizer):
    """Mengembalikan list (review_asli, label, clean_text) untuk setiap ulasan."""
    results = []
    for r in reviews:
        clean = preprocess_pipeline(r)
        if not clean:
            continue
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        results.append((r, int(pred), clean))
    return results


def detect_topics(clean_texts: list[str]) -> list[str]:
    """
    Mendeteksi topik (kata kunci utama) menggunakan pendekatan NLP (TF-IDF Keyword Extraction).
    Mengekstrak kata-kata yang paling penting secara statistik dari kumpulan teks.
    """
    if not clean_texts:
        return []

    # Gunakan TF-IDF untuk menemukan kata-kata kunci unik
    # Masukkan CUSTOM_STOPWORDS agar hasil ekstraksi lebih informatif
    local_vectorizer = TfidfVectorizer(
        max_features=15, 
        stop_words=CUSTOM_STOPWORDS
    )
    try:
        tfidf_matrix = local_vectorizer.fit_transform(clean_texts)
        feature_names = local_vectorizer.get_feature_names_out()
        
        # Hitung skor rata-rata TF-IDF untuk setiap kata
        scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Ambil top 3 kata kunci sebagai representasi "topik"
        top_indices = scores.argsort()[::-1][:3]
        topics = [feature_names[i] for i in top_indices if scores[i] > 0]
        
        return [t.capitalize() for t in topics]
    except:
        # Fallback jika teks terlalu sedikit atau semua kata ter-exclude
        combined = [w for w in " ".join(clean_texts).split() if w.lower() not in CUSTOM_STOPWORDS]
        if not combined: return []
        counts = Counter(combined)
        return [word.capitalize() for word, _ in counts.most_common(3)]


def render_verdict(positive_ratio: float, total: int):
    """Menampilkan kartu kesimpulan besar — bagian paling penting untuk user awam."""
    pct = round(positive_ratio * 100)

    if positive_ratio >= 0.75:
        color, emoji, label = "#1b873f", "👍", "Layak Dipertimbangkan"
        message = "Sebagian besar pembeli puas dengan produk/toko ini."
    elif positive_ratio >= 0.5:
        color, emoji, label = "#b8860b", "🤔", "Cukup Baik, Tetap Hati-hati"
        message = "Ada cukup banyak ulasan positif, tapi perhatikan keluhan di bawah."
    else:
        color, emoji, label = "#c0392b", "⚠️", "Pertimbangkan Ulang"
        message = "Lebih banyak pembeli yang kecewa. Sebaiknya cek toko/produk lain."

    st.markdown(
        f"""
        <div style="
            background-color: {color}15;
            border: 2px solid {color};
            border-radius: 14px;
            padding: 24px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="font-size: 42px;">{emoji}</div>
            <div style="font-size: 24px; font-weight: 700; color: {color};">{label}</div>
            <div style="font-size: 36px; font-weight: 800; margin: 8px 0; color: {color};">
                {pct}% pembeli puas
            </div>
            <div style="font-size: 14px; color: #444;">{message}</div>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">
                Berdasarkan analisis {total} ulasan
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# UI utama
# ------------------------------------------------------------------
st.title("🛍️ Asisten Belanja Pintar")
st.markdown(
    "Hai Bunda! Bingung mau belanja di mana? Pilih E-commerce di bawah untuk melihat "
    "apakah pengguna lain merasa puas bulan ini."
)

model, vectorizer, is_fallback = get_model()
df_full = load_dataset()

if df_full.empty:
    st.error("❌ Waduh, data ulasan tidak ditemukan. Pastikan file `data/dataset_gabungan.csv` ada ya!")
    st.stop()

# Ambil daftar marketplace unik dan rapikan namanya
available_apps = sorted(df_full['app_name'].unique().tolist())
app_display_names = {app: app.capitalize() for app in available_apps}

selected_app_label = st.selectbox(
    "👉 Pilih E-commerce yang ingin dicek:",
    options=available_apps,
    format_func=lambda x: app_display_names.get(x, x)
)

if selected_app_label:
    with st.spinner(f"Sedang merangkum ulasan {app_display_names[selected_app_label]}..."):
        # Predict seluruh ulasan platform — di-cache, jadi hanya dihitung sekali
        results = get_predictions_cached(selected_app_label)
        
        if not results:
            st.warning("Gagal menganalisis ulasan. Coba pilih yang lain ya Bun.")
        else:
            total = len(results)
            positives = [r for r in results if r[1] == 1]
            negatives = [r for r in results if r[1] == 0]
            pos_ratio = len(positives) / total
            
            st.divider()
            
            # 1. Persentase Kepuasan
            render_verdict(pos_ratio, total)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2. Kelebihan Utama (dari ulasan positif)
                st.subheader("🌟 Kelebihan Utama")
                if positives:
                    pos_topics = detect_topics([r[2] for r in positives])
                    if pos_topics:
                        for t in pos_topics:
                            st.success(f"**{t}**")
                    else:
                        st.write("Banyak yang suka, tapi topiknya campur-campur.")
                else:
                    st.write("Belum ada ulasan positif yang menonjol.")

            with col2:
                # 3. Keluhan Utama (dari ulasan negatif)
                st.subheader("🚩 Keluhan Utama")
                if negatives:
                    neg_topics = detect_topics([r[2] for r in negatives])
                    if neg_topics:
                        for t in neg_topics:
                            st.error(f"**{t}**")
                    else:
                        st.write("Ada keluhan, tapi topiknya beragam.")
                else:
                    st.write("Wah hebat! Hampir tidak ada keluhan.")

            # 4. Testimoni Nyata (Top 3 Positif & Negatif)
            st.divider()
            st.subheader("💬 Apa Kata Mereka?")
            
            t_col1, t_col2 = st.columns(2)
            
            with t_col1:
                st.markdown("**Ulasan Positif Terpopuler:**")
                if positives:
                    # Ambil 3 sampel (bisa ulasan terbaru atau acak, di sini kita ambil 3 dari list)
                    top_pos = positives[:3]
                    for i, (text, _, _) in enumerate(top_pos):
                        st.info(f"\"{text}\"")
                else:
                    st.write("Belum ada testimoni positif.")

            with t_col2:
                st.markdown("**Keluhan Pengguna:**")
                if negatives:
                    top_neg = negatives[:3]
                    for i, (text, _, _) in enumerate(top_neg):
                        st.warning(f"\"{text}\"")
                else:
                    st.write("Belum ada testimoni keluhan.")

            # Info tambahan mode demo jika aktif
            if is_fallback:
                st.caption("---")
                st.caption("⚠️ Catatan: Aplikasi berjalan dalam mode demo dengan data terbatas.")

st.divider()
st.caption(
    "Dibuat khusus untuk membantu Ibu-ibu memilih tempat belanja terbaik. "
    "Data diambil otomatis dari ribuan ulasan Play Store."
)
