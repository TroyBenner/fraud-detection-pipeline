import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline
import time

# =====================================================================
#  STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="FINRA Fraud Intelligence Engine",
    page_icon="💠",
    layout="wide"
)

# =====================================================================
#  MODEL & DATA LOADING FUNCTIONS
# =====================================================================

@st.cache_resource
def load_embed_model():
    """Loads the SentenceTransformer embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_explainer():
    """Loads FLAN-T5-small for simple article explanations."""
    return pipeline("text2text-generation", model="google/flan-t5-small")

@st.cache_data
def load_data():
    """Loads your processed FINRA CSV dataset."""
    df = pd.read_csv("fraud_analysis_final.csv")
    df["summary"] = df["summary"].astype(str)
    return df

embed_model = load_embed_model()
explain_model = load_explainer()
df = load_data()

@st.cache_resource
def embed_summaries(summaries):
    """Embeds all article summaries into dense vectors."""
    return embed_model.encode(summaries, convert_to_tensor=True)

summary_embeddings = embed_summaries(df["summary"].tolist())

# =====================================================================
#  FRAUD AUTO-TAGGING
# =====================================================================

FRAUD_TAGS = {
    "AI Fraud": ["ai", "artificial intelligence", "deepfake"],
    "Check Fraud": ["check fraud", "mail theft"],
    "Elder Fraud": ["older adults", "senior", "elderly"],
    "Account Takeover": ["account takeover", "hacked"],
    "Scams": ["scam", "romance", "crypto", "pump and dump"],
    "Disaster Fraud": ["natural disaster"],
    "General Fraud": ["fraud", "scheme", "securities"]
}

def classify_article(text):
    """Assigns each article a fraud category based on keywords."""
    text = text.lower()
    for label, words in FRAUD_TAGS.items():
        if any(w in text for w in words):
            return label
    return "General Fraud"

df["tag"] = df["summary"].apply(classify_article)

# =====================================================================
#  PREMIUM UI STYLING (CSS)
# =====================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0c111b;
    color: #f6f6f6;
}
.hero {
    font-size: 3.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #8b5cf6, #22d3ee, #6366f1);
    -webkit-background-clip: text;
    color: transparent;
    animation: glow 6s ease-in-out infinite;
}
@keyframes glow {0%{opacity:.85;}50%{opacity:1;}100%{opacity:.85;}}
.subtitle {
    text-align: center;
    opacity: .8;
    margin-bottom: 25px;
    font-size: 1.25rem;
}
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 28px;
    backdrop-filter: blur(12px);
    transition: all .3s ease;
    animation: fadein .8s;
}
.card:hover {
    transform: translateY(-6px);
    border-color: rgba(139,92,246,0.4);
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
}
@keyframes fadein {
    from {opacity: 0; transform: translateY(12px);}
    to {opacity: 1; transform: translateY(0);}
}
.chip {
    padding: 5px 14px;
    margin: 4px;
    font-size: .85rem;
    background: rgba(139,92,246,0.15);
    border-radius: 14px;
    border: 1px solid rgba(139,92,246,0.3);
    display: inline-block;
}
.divider {
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #22d3ee);
    margin: 20px 0;
    border-radius: 6px;
}
.sidebar-title {
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
#  SIDEBAR — SEARCH HISTORY + CLEAR BUTTON
# =====================================================================

st.sidebar.markdown("<p class='sidebar-title'>🔎 Search History</p>", unsafe_allow_html=True)

# initialize search history
if "history" not in st.session_state:
    st.session_state.history = []

# show search history
if len(st.session_state.history) > 0:
    for q in st.session_state.history[-10:]:
        st.sidebar.write(f"• {q}")
else:
    st.sidebar.write("No searches yet.")

# PREMIUM CLEAR HISTORY BUTTON
if st.sidebar.button("✨ Clear Search History"):
    st.session_state.history = []
    st.sidebar.success("History cleared!")

st.sidebar.markdown("---")

# fraud filter dropdown
tag_filter = st.sidebar.selectbox("Filter by fraud type", ["All"] + list(FRAUD_TAGS.keys()))

# =====================================================================
#  HERO HEADER
# =====================================================================

st.markdown("<h1 class='hero'>FINRA Fraud Intelligence Engine 💠</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Semantic Search • Article Intelligence • Fraud Classification • Compare Insights</p>", unsafe_allow_html=True)

# =====================================================================
#  SEARCH INPUT
# =====================================================================

query = st.text_input(
    "Ask a question:",
    placeholder="e.g., How does AI enable new investment fraud schemes?"
)

# =====================================================================
#  SEMANTIC SEARCH LOGIC
# =====================================================================

if query:
    # store in history
    st.session_state.history.append(query)

    with st.spinner("Analyzing your question…"):
        time.sleep(0.3)
        q_emb = embed_model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, summary_embeddings)[0]

        best_idx = torch.argmax(sims).item()
        best_score = float(sims[best_idx])
        best = df.iloc[best_idx]

    # filter mismatch warning
    if tag_filter != "All" and best["tag"] != tag_filter:
        st.info(f"No results match tag '{tag_filter}'. Showing top result instead.")

    # =================================================================
    #  RESULT CARD
    # =================================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(f"### 🏆 Best Match: **{best['title']}**")
    st.write(f"🔗 [Open Article]({best['url']})")
    st.write(f"**Relevance Score:** `{best_score:.4f}`")
    st.write(f"**Fraud Category:** `{best['tag']}`")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("### 📝 Summary")
    st.write(best["summary"])

    st.markdown("### 🔑 Keywords")
    if isinstance(best["keywords"], str):
        for kw in best["keywords"].split(","):
            st.markdown(f"<span class='chip'>{kw.strip()}</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    #  EXPLAIN MODE
    # =================================================================
    st.markdown("### 🤖 Explain This Article")
    if st.button("Generate Explanation"):
        with st.spinner("AI is simplifying the article…"):
            explanation = explain_model(
                f"Explain this in simple terms: {best['summary']}",
                max_length=180
            )[0]["generated_text"]
        st.success(explanation)

    # =================================================================
    #  COMPARE MODE
    # =================================================================
    st.markdown("### 📊 Compare With Another Article")

    compare_title = st.selectbox(
        "Select article to compare",
        options=df["title"].tolist()
    )

    if compare_title:
        other = df[df["title"] == compare_title].iloc[0]
        other_emb = embed_model.encode(other["summary"], convert_to_tensor=True)
        similarity = float(util.cos_sim(q_emb, other_emb)[0][0])

        st.write(f"**Similarity between your query and this article:** `{similarity:.4f}`")
