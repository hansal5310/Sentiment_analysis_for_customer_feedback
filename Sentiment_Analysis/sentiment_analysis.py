import streamlit as st
import pandas as pd
import pickle
import io
import re
import sqlite3
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ReviewSense",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Sora', sans-serif;
    background: #0d0f14;
    color: #e8eaf0;
}

.stApp {
    background: #0d0f14;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main container */
.block-container {
    padding: 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ---- HERO ---- */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6ee7b7;
    background: rgba(110,231,183,0.08);
    border: 1px solid rgba(110,231,183,0.2);
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    color: #f0f2f8;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(120deg, #6ee7b7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ---- DIVIDER ---- */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(110,231,183,0.15), transparent);
    margin: 2rem 0;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: #161920;
    border: 1px solid #1f2430;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    width: fit-content;
    margin: 0 auto 2.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #6b7280;
    font-family: 'Sora', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.55rem 1.4rem;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(110,231,183,0.15), rgba(56,189,248,0.15)) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(110,231,183,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- CARDS ---- */
.card {
    background: #161920;
    border: 1px solid #1f2430;
    border-radius: 16px;
    padding: 1.8rem;
    transition: border-color 0.2s ease;
}
.card:hover { border-color: rgba(110,231,183,0.2); }
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- TEXT AREA ---- */
.stTextArea textarea {
    background: #0d0f14 !important;
    border: 1px solid #1f2430 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: rgba(110,231,183,0.4) !important;
    box-shadow: 0 0 0 3px rgba(110,231,183,0.06) !important;
}
.stTextArea label { display: none !important; }

/* ---- BUTTON ---- */
.stButton > button {
    background: linear-gradient(135deg, #6ee7b7, #38bdf8) !important;
    color: #0d0f14 !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ---- SENTIMENT RESULT ---- */
.sentiment-result {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(110,231,183,0.05);
    border: 1px solid rgba(110,231,183,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.2rem;
}
.sentiment-emoji { font-size: 2rem; }
.sentiment-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
}
.sentiment-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: #6ee7b7;
}
.sentiment-neg { color: #f87171; border-color: rgba(248,113,113,0.15); background: rgba(248,113,113,0.05); }
.sentiment-neu { color: #fbbf24; border-color: rgba(251,191,36,0.15); background: rgba(251,191,36,0.05); }
.sentiment-pos { color: #6ee7b7; border-color: rgba(110,231,183,0.15); background: rgba(110,231,183,0.05); }

/* ---- METRICS ---- */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: #0d0f14;
    border: 1px solid #1f2430;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f2f8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ---- SELECT BOX ---- */
.stSelectbox > div > div {
    background: #0d0f14 !important;
    border: 1px solid #1f2430 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
}
.stSelectbox label { color: #6b7280!important; font-size: 0.8rem !important; }

/* ---- FILE UPLOADER ---- */
.stFileUploader > div {
    background: #0d0f14 !important;
    border: 1px dashed #1f2430 !important;
    border-radius: 10px !important;
}
.stFileUploader label { color: #6b7280 !important; }

/* ---- DOWNLOAD BUTTON ---- */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #1f2430 !important;
    color: #6ee7b7 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: rgba(110,231,183,0.4) !important;
    background: rgba(110,231,183,0.06) !important;
}

/* ---- DATAFRAME ---- */
.stDataFrame {
    border: 1px solid #1f2430 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ---- ALERTS ---- */
.stSuccess {
    background: rgba(110,231,183,0.08) !important;
    border: 1px solid rgba(110,231,183,0.2) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
.stError {
    background: rgba(248,113,113,0.08) !important;
    border: 1px solid rgba(248,113,113,0.2) !important;
    border-radius: 10px !important;
}
.stInfo {
    background: rgba(56,189,248,0.06) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 10px !important;
    color: #7dd3fc !important;
}

/* ---- FOOTER ---- */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #2d3344;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML-Powered · v2.0</div>
    <div class="hero-title">Review<span>Sense</span></div>
    <div class="hero-sub">Understand customer sentiment instantly — one review or thousands at once.</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -----------------------------
# PREPROCESSING
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

sentiment_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
sentiment_class = {0: "sentiment-neg", 1: "sentiment-neu", 2: "sentiment-pos"}
sentiment_emoji = {0: "😠", 1: "😐", 2: "😊"}
sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ======================================================
# TABS
# ======================================================
tab1, tab2 = st.tabs(["  Single Review  ", "  Bulk Scanner  "])

# ======================================================
# TAB 1 — MANUAL TEST
# ======================================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='card'><div class='card-title'>Input</div>", unsafe_allow_html=True)
        review = st.text_area("review", placeholder="Paste a customer review here…", height=180, label_visibility="collapsed")
        predict_btn = st.button("Analyze Sentiment →", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'><div class='card-title'>Result</div>", unsafe_allow_html=True)

        if predict_btn and review.strip():
            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            cls = sentiment_class.get(pred, "sentiment-pos")
            emoji = sentiment_emoji.get(pred, "")
            lbl = sentiment_label.get(pred, "")

            st.markdown(f"""
            <div class="sentiment-result {cls}">
                <div class="sentiment-emoji">{emoji}</div>
                <div>
                    <div class="sentiment-label">Predicted Sentiment</div>
                    <div class="sentiment-value">{lbl}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif predict_btn and not review.strip():
            st.info("Please enter a review first.")
        else:
            st.markdown("""
            <div style='color:#2d3344;text-align:center;padding:3rem 0;font-size:0.85rem;font-family:JetBrains Mono,monospace;'>
                — awaiting input —
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 2 — BULK SCANNER
# ======================================================
with tab2:
    sample_df = pd.DataFrame({
        "review_text": [
            "This product is amazing!",
            "Very bad quality, disappointed.",
            "Average experience, okay product."
        ]
    })

    col_a, col_b, col_c = st.columns(3, gap="large")

    # --- SAMPLE DOWNLOAD ---
    with col_a:
        st.markdown("<div class='card'><div class='card-title'>Sample Template</div>", unsafe_allow_html=True)
        fmt = st.selectbox("Format", ["CSV", "Excel", "JSON", "SQL"], label_visibility="visible")

        if fmt == "CSV":
            st.download_button("⬇ Download CSV", sample_df.to_csv(index=False), "sample_reviews.csv", "text/csv")
        elif fmt == "Excel":
            buf = io.BytesIO(); sample_df.to_excel(buf, index=False)
            st.download_button("⬇ Download Excel", buf.getvalue(), "sample_reviews.xlsx")
        elif fmt == "JSON":
            st.download_button("⬇ Download JSON", sample_df.to_json(orient="records", indent=2), "sample_reviews.json")
        elif fmt == "SQL":
            sql = """CREATE TABLE reviews (review_text TEXT);\n\nINSERT INTO reviews VALUES\n('This product is amazing!'),\n('Very bad quality, disappointed'),\n('Average experience, okay product');"""
            st.download_button("⬇ Download SQL", sql, "sample_reviews.sql")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- UPLOAD ---
    with col_b:
        st.markdown("<div class='card'><div class='card-title'>Upload File</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["csv","xlsx","json","sql"], label_visibility="collapsed")

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):     bulk_df = pd.read_csv(uploaded)
                elif uploaded.name.endswith(".xlsx"):  bulk_df = pd.read_excel(uploaded)
                elif uploaded.name.endswith(".json"):  bulk_df = pd.read_json(uploaded)
                elif uploaded.name.endswith(".sql"):
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(uploaded.read().decode("utf-8"))
                    bulk_df = pd.read_sql("SELECT * FROM reviews", conn)
                    conn.close()

                st.success(f"Loaded {len(bulk_df)} rows")
                st.dataframe(bulk_df.head(3), use_container_width=True)

                if st.button("Run Bulk Analysis →", use_container_width=True):
                    bulk_df["cleaned"] = bulk_df["review_text"].apply(clean_text)
                    vecs = vectorizer.transform(bulk_df["cleaned"])
                    bulk_df["predicted_sentiment"] = model.predict(vecs)
                    bulk_df["predicted_sentiment"] = bulk_df["predicted_sentiment"].map(sentiment_map)
                    st.session_state["bulk_result"] = bulk_df
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- RESULTS ---
    with col_c:
        st.markdown("<div class='card'><div class='card-title'>Results</div>", unsafe_allow_html=True)

        if "bulk_result" in st.session_state:
            res = st.session_state["bulk_result"]
            total = len(res)
            pos = (res["predicted_sentiment"] == "Positive 😊").sum()
            neg = (res["predicted_sentiment"] == "Negative 😠").sum()

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="metric-num">{total}</div><div class="metric-lbl">Total</div></div>
                <div class="metric-box"><div class="metric-num" style="color:#6ee7b7">{pos}</div><div class="metric-lbl">Positive</div></div>
                <div class="metric-box"><div class="metric-num" style="color:#f87171">{neg}</div><div class="metric-lbl">Negative</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(res[["review_text","predicted_sentiment"]].head(5), use_container_width=True)
            st.download_button("⬇ Download Results", res.to_csv(index=False), "predictions.csv", "text/csv")

        else:
            st.markdown("""
            <div style='color:#2d3344;text-align:center;padding:3rem 0;font-size:0.85rem;font-family:JetBrains Mono,monospace;'>
                — no results yet —
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="divider" style="margin-top:3rem"></div>
<div class="footer">REVIEWSENSE · ML-POWERED SENTIMENT ANALYSIS</div>
""", unsafe_allow_html=True)