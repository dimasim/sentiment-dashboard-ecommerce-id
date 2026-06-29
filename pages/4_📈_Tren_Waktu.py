"""
Halaman 4 — Tren Waktu
Line chart positive rate per bulan per platform (Nov 2023 – Mei 2026).
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from shared import PLATFORM_COLORS, PLATFORM_LABELS, load_dataset

st.set_page_config(page_title="Tren Waktu — Sentiment Dashboard", page_icon="📈", layout="wide")

st.title("📈 Tren Sentimen Waktu")
st.caption("Perubahan tingkat kepuasan pengguna dari bulan ke bulan per platform.")

df = load_dataset()

if "review_datetime" not in df.columns or "score" not in df.columns:
    st.error("Kolom `review_datetime` atau `score` tidak ditemukan.")
    st.stop()

# ── Persiapan data ────────────────────────────────────────────────────────────
df = df.copy()
df["year_month"] = df["review_datetime"].dt.to_period("M").dt.to_timestamp()
df["positif"] = (df["score"] >= 4).astype(int)

@st.cache_data(show_spinner="Menghitung tren...")
def compute_trend(_df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        _df.groupby(["year_month", "app_name"])
        .agg(total=("score", "count"), positif=("positif", "sum"))
        .reset_index()
    )
    trend["positive_rate"] = (trend["positif"] / trend["total"] * 100).round(2)
    trend["Platform"] = trend["app_name"].map(lambda x: PLATFORM_LABELS.get(x, x.capitalize()))
    return trend

trend_df = compute_trend(df)

# ── Filter platform ───────────────────────────────────────────────────────────
all_platforms = [PLATFORM_LABELS.get(p, p.capitalize()) for p in sorted(df["app_name"].unique())]
selected_platforms = st.multiselect(
    "Tampilkan Platform:",
    options=all_platforms,
    default=all_platforms,
)

if not selected_platforms:
    st.warning("Pilih minimal satu platform.")
    st.stop()

filtered_trend = trend_df[trend_df["Platform"].isin(selected_platforms)]

# ── Line Chart: Positive rate ─────────────────────────────────────────────────
st.subheader("Tingkat Kepuasan Bulanan (%)")

fig_line = px.line(
    filtered_trend,
    x="year_month",
    y="positive_rate",
    color="Platform",
    color_discrete_map={
        PLATFORM_LABELS.get(k, k.capitalize()): v for k, v in PLATFORM_COLORS.items()
    },
    markers=True,
    labels={"year_month": "Bulan", "positive_rate": "% Positif", "Platform": "Platform"},
)
fig_line.add_hline(y=50, line_dash="dash", line_color="gray",
                   annotation_text="50%", annotation_position="bottom right")
fig_line.update_layout(
    hovermode="x unified",
    yaxis=dict(range=[0, 100], title="% Ulasan Positif"),
    xaxis_title="Bulan",
    height=420,
    margin=dict(t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Area Chart: Volume ulasan per bulan ──────────────────────────────────────
st.subheader("Volume Ulasan per Bulan")

volume_df = (
    filtered_trend.groupby("year_month")
    .agg(total=("total", "sum"))
    .reset_index()
)

fig_area = px.area(
    volume_df,
    x="year_month",
    y="total",
    labels={"year_month": "Bulan", "total": "Jumlah Ulasan"},
    color_discrete_sequence=["#5B8FF9"],
)
fig_area.update_layout(
    height=280,
    margin=dict(t=20, b=20),
    xaxis_title="Bulan",
    yaxis_title="Jumlah Ulasan",
)
st.plotly_chart(fig_area, use_container_width=True)

st.caption(
    "📌 **Catatan**: Penurunan mendadak pada grafik volume bisa mengindikasikan "
    "data gap (periode tanpa ulasan yang terkumpul), bukan penurunan aktivitas pengguna yang nyata."
)
