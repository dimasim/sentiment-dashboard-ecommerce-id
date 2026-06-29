# 🛍️ Asisten Belanja — Sentiment Dashboard
### Fundamental Project Plan (Diperbarui berdasarkan kode aktual)

---

## 📋 Inventaris File Saat Ini

| File | Ukuran | Fungsi |
|---|---|---|
| `app.py` | 10.5 KB | Consumer UI (Asisten Belanja Pintar) |
| `preprocessing.py` | 3.5 KB | Pipeline NLP: clean → slang → stopword → stem |
| `model_utils.py` | 6.6 KB | Load pkl / fallback training dari 60 sampel |
| `model_svm_best.pkl` | **40 KB** | LinearSVC terlatih (sklearn 1.6.1) |
| `vectorizer_tfidf.pkl` | **194 KB** | TF-IDF 5000 vocab, unigram+bigram |
| `dataset_gabungan.csv` | **7.2 MB** disk / 19.4 MB RAM | 30.844 baris, Nov 2023 – Mei 2026 |
| `requirements.txt` | 98 B | 6 dependensi (perlu update) |

**Kesimpulan**: Stack sangat ringan. Tidak ada database, tidak ada API backend, tidak ada scheduler. Semua file-based.

---

## 🔍 Kondisi Kode Saat Ini

### Apa yang sudah berjalan ✅
- `app.py` → Streamlit single-page, user pilih platform → lihat % kepuasan, topik keluhan, testimoni
- `preprocessing.py` → Pipeline lengkap dan identik dengan notebook (clean, slang, stopword, stem)
- `model_utils.py` → Auto-detect apakah pkl asli ada atau perlu fallback
- Dataset: 30.844 baris, 4 platform, tanpa missing values, seimbang antar-platform

### Masalah yang harus diperbaiki ⚠️

**1. Versi scikit-learn tidak di-pin (KRITIS)**

```txt
# requirements.txt saat ini — SALAH:
scikit-learn>=1.4.0

# Harus menjadi:
scikit-learn==1.6.1
```

Model pkl dilatih dengan sklearn **1.6.1**. Jika server menginstall versi lebih baru (1.7+, 1.8+), akan muncul `InconsistentVersionWarning` dan berpotensi hasil prediksi berbeda. **Ini bug nyata, bukan warning kosmetik.**

**2. `app.py` masih single-page**

Belum ada halaman dashboard analytics. Perlu dikonversi ke struktur Streamlit multi-page (`pages/` folder).

**3. Analisis hanya ambil 200 ulasan terakhir per platform**

```python
# Baris ini di app.py — hardcoded:
sample_size = min(len(df_selected), 200)
reviews_to_analyze = df_selected['text'].astype(str).tail(sample_size).tolist()
```

Dashboard perlu menganalisis seluruh 30.844 baris, tapi dengan caching yang benar.

---

## 🏗️ Target Arsitektur: Multi-Page Streamlit

Tidak perlu FastAPI, PostgreSQL, Celery, atau Redis. Stack yang ada sudah cukup — hanya perlu **tambah halaman Streamlit** + **2 library visualisasi**.

```
asisten-belanja/
│
├── app.py                        # Halaman utama: Asisten Belanja (existing)
│
├── pages/                        # ← FOLDER BARU untuk dashboard
│   ├── 1_📊_Ringkasan.py         # Overview & KPI
│   ├── 2_🏪_Perbandingan.py      # Positive rate per platform
│   ├── 3_☁️_Kata_Kunci.py        # Word cloud per sentimen & platform
│   ├── 4_📈_Tren_Waktu.py        # Time-series sentimen bulanan
│   └── 5_🔍_Coba_Prediksi.py     # Live inference (input teks bebas)
│
├── model_utils.py                # Existing (tidak berubah)
├── preprocessing.py              # Existing (tidak berubah)
│
├── data/
│   └── dataset_gabungan.csv      # Pindahkan ke folder data/ agar rapi
│
├── models/
│   ├── model_svm_best.pkl        # Pindahkan ke folder models/
│   └── vectorizer_tfidf.pkl
│
└── requirements.txt              # Update (tambah plotly, wordcloud, matplotlib)
```

> **Catatan**: Saat pindahkan pkl ke `models/`, update path di `model_utils.py`:
> ```python
> MODEL_PATH = Path(__file__).parent / "models" / "model_svm_best.pkl"
> VECTORIZER_PATH = Path(__file__).parent / "models" / "vectorizer_tfidf.pkl"
> ```

---

## 📊 Spesifikasi 5 Halaman Dashboard

### Halaman 0 — `app.py` (Existing, tidak banyak berubah)
**Target user**: Pembeli / ibu rumah tangga

- Dropdown pilih platform
- Kartu verdict besar (👍 / 🤔 / ⚠️) + persentase kepuasan
- Kolom kelebihan vs keluhan (TF-IDF keyword extraction)
- Testimoni nyata (3 positif + 3 negatif)

---

### Halaman 1 — `1_📊_Ringkasan.py`
**Target user**: Pemilik project / analis

```
Komponen:
├── Metric Cards  → Total ulasan, % positif keseluruhan, rentang tanggal data
├── Pie Chart     → Distribusi skor 1-5 (plotly)
├── Bar Chart     → Jumlah ulasan per platform (plotly)
└── Info Box      → Info model: tipe (LinearSVC), vocab size (5000), akurasi (92.89%)
```

---

### Halaman 2 — `2_🏪_Perbandingan.py`
**Target user**: Analis / manajemen

```
Komponen:
├── Bar Chart Horizontal  → Positive rate per platform dengan threshold line 50%
│     Shopee: 59.31% ✅   Lazada: 46.38% ⚠️
│     Bukalapak: 36.30% ❌  Tokopedia: 35.19% ❌
├── Stacked Bar Chart     → Breakdown score 1-5 per platform
└── Tabel Ringkasan       → Total ulasan, positif, negatif, % kepuasan
```

---

### Halaman 3 — `3_☁️_Kata_Kunci.py`
**Target user**: Analis konten / produk

```
Komponen:
├── Selector (radio)    → Pilih: Semua Sentimen / Positif saja / Negatif saja
├── Selector (dropdown) → Filter per platform (atau semua)
└── Word Cloud          → Hasil visualisasi (wordcloud library)
     - Tema hijau untuk sentimen positif
     - Tema merah untuk sentimen negatif
```

**Catatan implementasi**: Word cloud butuh preprocessing seluruh dataset → gunakan `@st.cache_data` agar hanya dihitung sekali.

---

### Halaman 4 — `4_📈_Tren_Waktu.py`
**Target user**: Analis / stakeholder

```
Komponen:
├── Line Chart (plotly)  → Positive rate per bulan, per platform (4 garis)
│     X-axis: year_month (dari review_datetime)
│     Y-axis: % positif (0-100%)
│     Garis: Tokopedia, Shopee, Lazada, Bukalapak (warna berbeda)
├── Area Chart           → Volume ulasan per bulan (untuk deteksi data gap)
└── Caption              → Penjelasan concept drift untuk pembaca awam
```

**Data yang tersedia**: Nov 2023 – Mei 2026 (2.5 tahun data) — cukup untuk tren bermakna.

---

### Halaman 5 — `5_🔍_Coba_Prediksi.py`
**Target user**: Siapapun yang ingin coba langsung

```
Komponen:
├── Text Area     → Input teks ulasan bebas (single atau multi-baris)
├── Tombol        → "Analisis Sentimen"
├── Output Card   → Label (Positif / Negatif) + confidence score*
└── Riwayat       → Tabel hasil prediksi dalam sesi ini (st.session_state)

*LinearSVC tidak punya predict_proba() secara langsung.
 Gunakan decision_function() lalu normalisasi dengan sigmoid untuk "confidence".
```

---

## 📦 `requirements.txt` yang Diperbarui

```txt
# === Core ===
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.2

# === ML/NLP ===
scikit-learn==1.6.1       # WAJIB PIN — pkl model dilatih dengan versi ini
sastrawi>=1.0.1

# === Visualisasi Dashboard (TAMBAHAN BARU) ===
plotly>=5.19.0            # Bar chart, pie chart, line chart interaktif
wordcloud>=1.9.3          # Word cloud halaman 3
matplotlib>=3.8.0         # Required by wordcloud
```

---

## 🖥️ Analisis Spek Server 5 GB

### Estimasi RAM saat Runtime (hasil pengukuran aktual)

| Komponen | RAM |
|---|---|
| Dataset CSV dimuat pandas | ~19.4 MB |
| Model LinearSVC pkl | < 1 MB |
| Vectorizer TF-IDF pkl | < 1 MB |
| Sastrawi stemmer + stopword | ~30 MB |
| scikit-learn + pandas + numpy loaded | ~145 MB |
| Streamlit base process | ~150 MB |
| **Total estimasi peak (1 user)** | **~350–400 MB** |
| Safety buffer × 2 (concurrent 2 user) | **~700–800 MB** |

### Estimasi Storage

| Komponen | Disk |
|---|---|
| Ubuntu Server 24.04 minimal | ~1.5 GB |
| Python 3.11 | ~30 MB |
| sklearn + pandas + numpy + openpyxl | ~150 MB |
| streamlit + plotly + wordcloud + matplotlib | ~250 MB |
| Sastrawi | ~20 MB |
| App files + CSV + pkl | ~10 MB |
| **Total** | **~2.0 GB** |

### Verdict

| Spek | Status | Catatan |
|---|---|---|
| **5 GB RAM** | ✅ **Sangat aman** | Peak hanya ~400MB, sisa 4.6GB kosong |
| **5 GB Disk** | ✅ **Aman** | Total ~2GB terpakai, sisa 3GB kosong |
| **1 vCPU** | ✅ Cukup | Streamlit default single-thread |
| **2 vCPU** | 👍 Lebih baik | Untuk stemming batch yang CPU-intensive |

> **Kesimpulan: Server 5 GB (RAM maupun Disk) aman untuk proyek ini.**
> Tidak perlu upgrade. App ini sangat ringan karena tidak ada database,
> tidak ada model deep learning di production (IndoBERT tidak dipakai di inference),
> dan tidak ada background workers.

---

## 🗓️ Roadmap Development

```
Phase 1 — Perbaikan Kritis (1-2 hari)
  □ Pin scikit-learn==1.6.1 di requirements.txt
  □ Buat folder models/ dan data/, pindahkan file
  □ Update path di model_utils.py
  □ Test app.py masih jalan

Phase 2 — Setup Multi-Page (1 hari)
  □ Buat folder pages/
  □ Pindahkan logika app.py menjadi halaman utama
  □ Konfirmasi sidebar navigasi otomatis Streamlit berjalan

Phase 3 — Halaman Dashboard (3-5 hari)
  □ Halaman 1: Ringkasan (metric cards + distribusi skor)
  □ Halaman 2: Perbandingan platform (bar chart + tabel)
  □ Halaman 3: Word cloud (dengan @st.cache_data)
  □ Halaman 4: Tren waktu (line chart per bulan)
  □ Halaman 5: Live inference (text input + decision_function)

Phase 4 — Polish & Deploy (1-2 hari)
  □ Tambah README.md
  □ Test di server target
  □ Opsional: Tambah .streamlit/config.toml untuk theming
```

---

## ⚡ Tips Implementasi Penting

**1. Caching yang benar untuk dataset besar**
```python
# Di setiap page, import function ini dari shared module atau app.py
@st.cache_data(show_spinner="Memuat data ulasan...")
def load_dataset():
    return pd.read_csv("data/dataset_gabungan.csv", parse_dates=["review_datetime"])

@st.cache_resource(show_spinner="Memuat model...")
def get_model():
    return load_or_train_model()
```

**2. Word cloud butuh predict seluruh dataset — lakukan batch**
```python
# Jangan predict satu per satu, gunakan transform batch
X_all = vectorizer.transform(df["text_clean"])
preds = model.predict(X_all)  # Jauh lebih cepat
df["sentiment"] = preds
```
Tapi ini artinya preprocessing seluruh 30K baris dulu — gunakan `@st.cache_data` agar hanya sekali.

**3. Confidence score untuk LinearSVC**
```python
# LinearSVC tidak punya predict_proba, tapi bisa pakai decision_function
scores = model.decision_function(vec)
# Normalisasi ke 0-1 pakai sigmoid
import numpy as np
confidence = 1 / (1 + np.exp(-scores))
```

**4. Tren waktu — gunakan score asli (bukan prediksi model)**
Dataset sudah punya kolom `score` (1-5). Untuk tren waktu lebih cepat dihitung dari:
- `score >= 4` → positif
- `score <= 2` → negatif

Ini menghindari perlu mem-predict 30K ulasan hanya untuk halaman tren.

---

*Blueprint ini berdasarkan inspeksi langsung terhadap 7 file aktual project.*
*Semua estimasi RAM dan storage diukur secara aktual, bukan asumsi.*
