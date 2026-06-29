# 🛍️ Sentiment Dashboard E-commerce Indonesia

Dashboard analisis sentimen ulasan pengguna dari 4 platform e-commerce Indonesia: **Shopee, Tokopedia, Lazada, dan Bukalapak**.

Dibangun dengan **Streamlit + LinearSVC (scikit-learn)** dan dataset 30.844 ulasan (Nov 2023 – Mei 2026).

---

## ✨ Fitur

| Halaman | Deskripsi |
|---|---|
| 🛍️ Asisten Belanja | Verdict kepuasan per platform untuk pengguna awam |
| 📊 Ringkasan | KPI cards, distribusi skor, info model |
| 🏪 Perbandingan | Positive rate & breakdown skor antar platform |
| ☁️ Kata Kunci | Word cloud interaktif per sentimen & platform |
| 📈 Tren Waktu | Line chart kepuasan bulanan per platform |
| 🔍 Coba Prediksi | Live inference teks bebas + confidence score |

---

## 🚀 Cara Jalankan

### Menggunakan Docker (Rekomendasi)

```bash
# 1. Clone repo
git clone https://github.com/dimasim/sentiment-dashboard-ecommerce-id.git
cd sentiment-dashboard-ecommerce-id

# 2. Salin env file
cp .env.example .env

# 3. Build & jalankan
docker compose up -d --build

# 4. Buka browser
# http://localhost:8501
```

**Perintah Docker berguna:**
```bash
docker compose logs -f          # Lihat log
docker compose restart          # Restart container
docker compose down             # Stop & hapus container
docker compose up -d --build    # Rebuild setelah ada perubahan kode
```

### Tanpa Docker (Manual)

```bash
# Python 3.11+ diperlukan
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ Konfigurasi `.env`

Salin `.env.example` ke `.env` dan sesuaikan:

```env
STREAMLIT_PORT=8501    # Port yang diekspos ke host
APP_ENV=production
TZ=Asia/Jakarta
```

---

## 🗂️ Struktur Proyek

```
sentiment-dashboard-ecommerce-id/
│
├── app.py                        # Halaman utama: Asisten Belanja
├── shared.py                     # Shared utilities (load data, model)
├── preprocessing.py              # Pipeline NLP (clean → slang → stem)
├── model_utils.py                # Load model pkl / fallback training
│
├── pages/
│   ├── 1_📊_Ringkasan.py
│   ├── 2_🏪_Perbandingan.py
│   ├── 3_☁️_Kata_Kunci.py
│   ├── 4_📈_Tren_Waktu.py
│   └── 5_🔍_Coba_Prediksi.py
│
├── data/
│   └── dataset_gabungan.csv      # 30.844 ulasan, 4 platform
│
├── models/
│   ├── model_svm_best.pkl        # LinearSVC terlatih (sklearn 1.6.1)
│   └── vectorizer_tfidf.pkl      # TF-IDF 5000 vocab
│
├── .streamlit/
│   └── config.toml               # Tema & konfigurasi Streamlit
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## 🧠 Model

- **Algoritma**: LinearSVC
- **Fitur**: TF-IDF (5000 vocab, unigram + bigram)
- **Preprocessing**: Cleaning → Slang normalization → Stopword removal → Sastrawi stemming
- **Akurasi**: 92.89% pada test set
- **Scikit-learn**: `==1.6.1` (di-pin untuk konsistensi pkl)

---

## 📦 Requirements Utama

```
streamlit>=1.30.0
scikit-learn==1.6.1
pandas>=2.0.0
plotly>=5.19.0
wordcloud>=1.9.3
sastrawi>=1.0.1
```

---

## 🖥️ Spek Server Minimum

| Komponen | Minimum | Rekomendasi |
|---|---|---|
| RAM | 1 GB | 2 GB |
| Disk | 3 GB | 5 GB |
| CPU | 1 vCPU | 2 vCPU |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

> Peak RAM usage ~400MB untuk 1 user aktif.

---

*Tugas Besar AI — Analisis Sentimen Ulasan E-commerce Indonesia*
