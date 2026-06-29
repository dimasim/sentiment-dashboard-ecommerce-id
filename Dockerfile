# ============================================================
# Dockerfile — Sentiment Dashboard E-commerce Indonesia
# Base: Python 3.11 slim (ringan, ~150MB)
# ============================================================
FROM python:3.11-slim

# Set timezone dan locale
ENV TZ=Asia/Jakarta \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit — nonaktifkan telemetry & browser auto-open
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# Install system deps yang dibutuhkan wordcloud & matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies dulu (layer terpisah → cache lebih efisien)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

# Port default Streamlit
EXPOSE 8501

# Health check — pastikan app merespons
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Jalankan Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
