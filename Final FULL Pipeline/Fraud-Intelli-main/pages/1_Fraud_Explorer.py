import streamlit as st
import pandas as pd
from intellifraud_ui import inject_light_ui
from load_data_supabase import load_fraud_data

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Fraud Explorer", layout="wide")
inject_light_ui()

# ------------------------------------------------------------
# LOGO AT TOP
# ------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:10px;">
        <img src="https://i.imgur.com/lAVJ7Vx.png" width="200">
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# BULLET-PROOF CSS FOR VERY LIGHT GRAY SELECTBOX + BLACK TEXT
# ------------------------------------------------------------
st.markdown("""
<style>

/* Full select container */
div[data-baseweb="select"] {
    background-color: #F8F9FA !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: 1px solid #D1D5DB !important;
}

/* Outer wrapper (ensures background stays gray) */
.stSelectbox > div {
    background-color: #F8F9FA !important;
}

/* Placeholder text */
div[data-baseweb="select"] .css-1wa3eu0-placeholder,
div[data-baseweb="select"] div[class*="placeholder"] {
    color: #000 !important;
    opacity: 0.7 !important;
}

/* Selected value text */
div[data-baseweb="select"] .css-1uccc91-singleValue,
div[data-baseweb="select"] div[class*="singleValue"] {
    color: #000 !important;
    opacity: 1 !important;
}

/* Dropdown menu background */
ul[role="listbox"] {
    background-color: #F8F9FA !important;
    border-radius: 6px !important;
}

/* Dropdown option text */
ul[role="listbox"] li {
    color: #000 !important;
    background-color: #F8F9FA !important;
}

/* Hover behavior */
ul[role="listbox"] li:hover {
    background-color: #E1E5EA !important;
    color: #000 !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD DATA FROM SUPABASE
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = load_fraud_data()
    return df

df = load_data()

# ------------------------------------------------------------
# FRAUD CATEGORIES
# ------------------------------------------------------------
FRAUD_CATEGORIES = {
    "Investment Fraud": [
        "investment fraud", "investment scams", "investment scam", "investing",
        "invest ai", "ai trading", "ai investment"
    ],
    "Financial Fraud": [
        "financial fraud", "financial scam", "fraud recovering",
        "fraud awareness", "fraud specialists", "fraud trends"
    ],
    "Cyber & AI Fraud": [
        "genai fraud", "accounts genai", "computer fraudsters",
        "ai trading", "ai investment", "artificial fraud"
    ],
    "Mail & Check Fraud": [
        "check fraud", "stolen checks", "mail theft", "mail check", "mail fraud"
    ],
    "Elder Fraud": [
        "older adults", "increase fraud", "fraud exposure", "fraud risk"
    ],
    "Disaster & Emergency Fraud": [
        "disaster fraud", "natural disasters", "disaster contribute"
    ],
    "Money Laundering": [
        "money laundering", "illicit finance", "laundering fraud"
    ],
}

# Normalize keywords
for cat in FRAUD_CATEGORIES:
    FRAUD_CATEGORIES[cat] = [kw.lower() for kw in FRAUD_CATEGORIES[cat]]

# ------------------------------------------------------------
# MATCHING FUNCTIONS
# ------------------------------------------------------------
def match_score(article_keywords, category_keywords):
    score = 0
    for a_kw in article_keywords:
        for c_kw in category_keywords:
            if c_kw in a_kw or a_kw in c_kw:
                score += 1
    return score

def get_articles(category_name):
    cat_keywords = FRAUD_CATEGORIES[category_name]
    results = []

    for _, row in df.iterrows():
        keywords_list = row["keywords"] if isinstance(row["keywords"], list) else []
        score = match_score(keywords_list, cat_keywords)

        if score > 0:
            results.append((score, row))

    results.sort(key=lambda x: x[0], reverse=True)
    return [row for score, row in results]

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div style="
    padding: 25px;
    background: #F5F7FA;
    border-radius: 15px;
    border: 1px solid #E6E9EF;
    margin-bottom: 20px;
">
    <h1 style="text-align:center; color:#0A1A2F;">🗂️ Fraud Explorer</h1>
    <p style="text-align:center; font-size:17px; color:#0A1A2F;">
        Explore regulatory enforcement articles organized by real-world fraud categories.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CATEGORY PICKER
# ------------------------------------------------------------
st.subheader("🧭 Select a Fraud Category")

selected_category = st.selectbox(
    "Choose a category to explore:",
    list(FRAUD_CATEGORIES.keys())
)

st.markdown(
    f"<h2 style='color:#0A65FF; margin-top:15px;'>📂 {selected_category}</h2>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# DISPLAY MATCHING ARTICLES
# ------------------------------------------------------------
articles = get_articles(selected_category)

if not articles:
    st.warning("No matching articles found for this category yet.")
else:
    for row in articles:
        title = row["title"]
        summary = row["summary"]
        url = row["url"]
        keywords = ", ".join(row["keywords"]) if isinstance(row["keywords"], list) else ""

        st.markdown(f"""
        <div style="
            background-color:#FFFFFF;
            padding:18px;
            margin-bottom:15px;
            border-radius:12px;
            border:1px solid #E6E9EF;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);
        ">
            <h3 style="color:#0A1A2F; margin-bottom:6px;">{title}</h3>
            <p style="color:#0A1A2F;">{summary}</p>
            <p><strong>Keywords:</strong> {keywords}</p>
            <a href="{url}" target="_blank" style="color:#0A65FF; font-weight:600;">
                Read Full Article →
            </a>
        </div>
        """, unsafe_allow_html=True)
