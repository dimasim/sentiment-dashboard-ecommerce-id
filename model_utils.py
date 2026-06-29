"""
Modul untuk melatih model fallback (SVM + TF-IDF) jika belum ada model
hasil training asli (model_svm_best.pkl) yang di-load dari notebook penelitian.

Begitu kamu punya file model_svm_best.pkl dan vectorizer_tfidf.pkl hasil
notebook (Colab), letakkan di folder ini -> app otomatis memakainya
dan modul ini tidak akan dipanggil lagi.
"""
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocessing import preprocess_pipeline

# Model asli (dari Colab) bisa dilatih dengan versi scikit-learn yang
# sedikit berbeda dari versi lokal. Ini aman untuk model klasik seperti
# LinearSVC/TfidfVectorizer, jadi warning-nya disembunyikan agar tidak
# membingungkan pengguna awam di UI.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_PATH = Path(__file__).parent / "models" / "model_svm_best.pkl"
VECTORIZER_PATH = Path(__file__).parent / "models" / "vectorizer_tfidf.pkl"

# Data contoh ulasan marketplace (positif & negatif) untuk fallback training.
# Mewakili pola bahasa gaul/informal khas ulasan e-commerce Indonesia.
SAMPLE_REVIEWS = [
    # Positif
    ("barang nya bagus banget sesuai sama foto, pengiriman juga cepet", 1),
    ("mantap seller nya ramah, packing rapi, recommended banget deh", 1),
    ("aku puas belanja disini, kualitas oke harga juga murah", 1),
    ("pengiriman cepet bgt, ga nyangka, terima kasih ya kak", 1),
    ("produk original, sesuai deskripsi, cs nya juga fast respon", 1),
    ("udah langganan disini terus, ga pernah kecewa, top deh", 1),
    ("kemasan aman, barang ga lecet sama sekali, mantul lah", 1),
    ("respon cepat, pengiriman aman, kualitas bagus, sangat puas", 1),
    ("akhirnya nemu toko yang amanah, barang sesuai pesanan", 1),
    ("ini udah ke 3 kalinya order disini, selalu memuaskan", 1),
    ("seller fast respon, barang sampai lebih cepat dari estimasi", 1),
    ("kualitas bahan nya bagus, ga murahan, worth it banget", 1),
    ("pelayanan ramah, harga bersahabat, pasti order lagi", 1),
    ("aplikasi nya mudah dipakai, belanja jadi lebih praktis", 1),
    ("barangnya rapi banget packingnya, ga ada yang rusak", 1),
    ("sangat puas, sesuai ekspektasi, terima kasih banyak", 1),
    ("ukurannya pas, bahannya nyaman dipakai, recommended", 1),
    ("toko terpercaya, responnya cepat, barang sesuai", 1),
    ("pengalaman belanja yang sangat menyenangkan, lanjut order lagi", 1),
    ("kualitas premium dengan harga terjangkau, suka banget", 1),
    ("bagus sekali barangnya, packing aman sentosa, seller amanah", 1),
    ("harga paling murah dibanding toko sebelah, barang tetep ori", 1),
    ("cepet banget sampai nya cuma sehari doang, makasih", 1),
    ("barangnya sesuai keinginan, ga ada cacat, berfungsi baik", 1),
    ("sukaa banget sama modelnya, elegan dan mewah", 1),
    ("pengiriman super kilat, barang sesuai deskripsi, mantap", 1),
    ("pelayanan oke, barang oke, harga oke, langganan deh", 1),
    ("adminnya baik banget, dibantu pas ada kendala, makasih ya", 1),
    ("packaging sangat eksklusif, aman sampai tujuan", 1),
    ("best seller pokoknya, barang bagus harga bersaing", 1),
    # Negatif
    ("barang nya jelek banget ga sesuai sama foto, kecewa", 0),
    ("pengiriman lama banget udah seminggu belum nyampe", 0),
    ("produk rusak waktu sampai, packing nya juga asal asalan", 0),
    ("seller nya susah dihubungi, cs ga respon sama sekali", 0),
    ("barang ga sesuai pesanan, ukuran beda, kecewa berat", 0),
    ("kualitas murahan, ga sesuai harga, nyesel beli disini", 0),
    ("pesenan ga lengkap, ada yang ketinggalan, parah banget", 0),
    ("barang nya cacat, ada yang sobek, mohon ganti", 0),
    ("lama banget responnya, udah komplain ga ada solusi", 0),
    ("ga sesuai deskripsi, warna beda jauh, sangat kecewa", 0),
    ("packing nya jelek banget barang sampai udah penyok", 0),
    ("pengiriman lambat dan barang yang dikirim salah", 0),
    ("kualitas buruk, baru dipake sekali udah rusak", 0),
    ("tidak sesuai gambar, kecewa berat, jangan beli disini", 0),
    ("pelayanan buruk, komplain ga ditanggapi, mengecewakan", 0),
    ("barang yang sampai berbeda total sama yang dipesan", 0),
    ("zonk banget, mohon di refund, sangat tidak puas", 0),
    ("ga rekomen, kualitas jelek, harga ga sesuai value", 0),
    ("pengalaman belanja paling buruk, ga akan order lagi", 0),
    ("aplikasi sering error, susah banget mau checkout", 0),
    ("kapok beli disini, barang bekas dikirim, parah", 0),
    ("nyesel banget, barangnya baru dipake bentar udah jebol", 0),
    ("pelayanan tidak memuaskan, chat cuma di read doang", 0),
    ("ongkir mahal tapi nyampenya lama banget, mengecewakan", 0),
    ("jangan mau beli disini, barang tidak sesuai iklan", 0),
    ("penipu ini mah, barang ga dikirim kirim", 0),
    ("deskripsi menyesatkan, kirain dapet 2 ternyata cuma 1", 0),
    ("kurirnya ga sopan, barang ditaruh sembarangan", 0),
    ("refund ribet banget, udah sebulan ga balik duitnya", 0),
    ("kecewa sekali, barang pecah pas dibuka, packing tipis", 0),
]


def _train_fallback_model():
    """Melatih model SVM + TF-IDF sederhana dari data sample."""
    df = pd.DataFrame(SAMPLE_REVIEWS, columns=["text", "label"])
    df["text_clean"] = df["text"].apply(preprocess_pipeline)

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text_clean"])
    y = df["label"]

    model = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer


def load_or_train_model():
    """
    Memuat model dari file pickle jika tersedia (hasil notebook asli).
    Jika belum ada, latih model fallback dari data sample sebagai demo.
    Mengembalikan tuple (model, vectorizer, is_fallback: bool).
    """
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        # Deteksi apakah ini model fallback (dilatih dari 40 sample) atau model asli.
        is_fallback = getattr(model, "_is_fallback_marker", False)
        return model, vectorizer, is_fallback

    model, vectorizer = _train_fallback_model()
    model._is_fallback_marker = True
    return model, vectorizer, True
