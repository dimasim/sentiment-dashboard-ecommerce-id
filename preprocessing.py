"""
Modul preprocessing teks ulasan Bahasa Indonesia.
Pipeline ini identik dengan yang digunakan pada notebook penelitian:
cleaning -> normalisasi slang -> stopword removal -> stemming (Sastrawi).
"""
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi tools Sastrawi (sekali saja, di-cache oleh Streamlit) ---
_stemmer = StemmerFactory().create_stemmer()
_stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# --- Kamus slang Indonesia (sama dengan notebook) ---
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "gk": "tidak", "tdk": "tidak",
    "dgn": "dengan", "yg": "yang", "krn": "karena", "karna": "karena",
    "bgt": "banget", "bngtt": "banget", "bener": "benar", "bner": "benar",
    "udh": "sudah", "udah": "sudah", "blm": "belum", "blom": "belum",
    "jg": "juga", "jgn": "jangan", "tp": "tapi", "tpi": "tapi",
    "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana",
    "knp": "kenapa", "knpa": "kenapa", "emg": "memang", "emang": "memang",
    "org": "orang", "orng": "orang", "skrg": "sekarang", "skrng": "sekarang",
    "skg": "sekarang", "lgsg": "langsung", "lsg": "langsung", "sbg": "sebagai",
    "utk": "untuk", "klo": "kalau", "kalo": "kalau", "dl": "dulu", "dlu": "dulu",
    "bs": "bisa", "bsa": "bisa", "trs": "terus", "trus": "terus",
    "jd": "jadi", "jdi": "jadi", "sy": "saya", "gw": "saya", "gue": "saya",
    "aku": "saya", "lo": "kamu", "lu": "kamu", "loe": "kamu", "ente": "kamu",
    "kuy": "yuk", "mantap": "bagus", "mantul": "bagus", "keren": "bagus",
    "jelek": "buruk", "zonk": "buruk",
    "aplikasi": "app", "aplikasinya": "app", "apk": "app",
}


def cleaning_text(text: str) -> str:
    """Lowercase, hapus URL, mention, hashtag, angka, dan tanda baca."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str) -> str:
    """Normalisasi kata slang ke bentuk baku menggunakan kamus."""
    words = text.split()
    return " ".join(SLANG_DICT.get(w, w) for w in words)


def remove_stopwords(text: str) -> str:
    return _stopword_remover.remove(text)


def stem_text(text: str) -> str:
    return _stemmer.stem(text)


def preprocess_pipeline(text: str) -> str:
    """Pipeline lengkap: cleaning -> slang -> stopword -> stemming."""
    text = cleaning_text(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text


def split_reviews(raw_text: str) -> list[str]:
    """
    Memecah teks mentah (paste user) menjadi list ulasan terpisah.
    Mendukung pemisah baris baru ATAU titik/koma jika user paste dalam satu paragraf panjang.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    # Jika hanya 1 baris panjang, coba pecah berdasarkan kalimat
    if len(lines) <= 1 and lines and len(lines[0]) > 200:
        candidate = re.split(r"(?<=[.!?])\s+", lines[0])
        candidate = [c.strip() for c in candidate if len(c.strip()) >= 3]
        if len(candidate) > 1:
            return candidate

    return [l for l in lines if len(l) >= 3]
