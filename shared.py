"""
Shared utilities untuk semua halaman dashboard.
Import dari sini agar tidak duplikasi kode load dataset & model.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import load_or_train_model

DATA_PATH = Path(__file__).parent / "data" / "dataset_gabungan.csv"

PLATFORM_COLORS = {
    "shopee": "#EE4D2D",
    "tokopedia": "#00AA5B",
    "lazada": "#0F146D",
    "bukalapak": "#D6001C",
}

PLATFORM_LABELS = {
    "shopee": "Shopee",
    "tokopedia": "Tokopedia",
    "lazada": "Lazada",
    "bukalapak": "Bukalapak",
}


@st.cache_data(show_spinner="Memuat data ulasan...")
def load_dataset() -> pd.DataFrame:
    """Load CSV sekali, cache selamanya selama sesi."""
    df = pd.read_csv(DATA_PATH, parse_dates=["review_datetime"], low_memory=False)
    # Normalisasi nama platform ke lowercase
    df["app_name"] = df["app_name"].str.lower().str.strip()
    return df


@st.cache_resource(show_spinner="Memuat model...")
def get_model():
    """Load model & vectorizer sekali, cache sebagai resource."""
    return load_or_train_model()


def sentiment_color(label: str) -> str:
    """Kembalikan warna hex berdasarkan label sentimen."""
    return "#1b873f" if label == "Positif" else "#c0392b"
