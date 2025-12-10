import streamlit as st
import pandas as pd
import altair as alt
import itertools
from pyvis.network import Network
import streamlit.components.v1 as components

from intellifraud_ui import inject_light_ui, sidebar_logo
from load_data_supabase import load_fraud_data

# ---------------------------------------------
# PAGE CONFIG & UI
# ---------------------------------------------
st.set_page_config(page_title="Fraud Trends", layout="wide")
inject_light_ui()

# ---------------------------------------------
# MOVE LOGO TO TOP OF PAGE
# ---------------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:15px; margin-bottom:20px;">
    <img src="https://i.imgur.com/lAVJ7Vx.png" width="160" style="border-radius:12px;">
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------
# HEADER HERO
# ---------------------------------------------
st.markdown("""
<div style="
    padding: 25px; 
    background: #F5F7FA; 
    border-radius: 15px; 
    border: 1px solid #E6E9EF; 
    margin-bottom: 20px;
    text-align:center;
">
    <h1 style="color:#0A1A2F; margin-bottom:5px;">
        📊 Fraud Trends & Analytics
    </h1>
    <p style="font-size:17px; color:#0A1A2F;">
        Explore trends in regulatory enforcement, keyword activity, and historical fraud patterns.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
@st.cache_data
def load_data():
    df = load_fraud_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def ensure_list(val):
        if isinstance(val, list):
            return val
        try:
            return eval(val)
        except:
            return []
    df["keywords"] = df["keywords"].apply(ensure_list)
    return df

df = load_data()

# ---------------------------------------------
# KEYWORD PROCESSING
# ---------------------------------------------
all_keywords = [kw for lst in df["keywords"] for kw in lst]
keyword_freq = pd.Series(all_keywords).value_counts().reset_index()
keyword_freq.columns = ["keyword", "count"]

# =====================================================
# SECTION 0 — TOP KEYWORDS
# =====================================================
st.subheader("🏆 Top Fraud Keywords")
st.markdown("<p style='color:#0A1A2F;'>Most common fraud-related keywords.</p>", unsafe_allow_html=True)

for _, row in keyword_freq.head(10).iterrows():
    st.markdown(f"""
    <div style="
        padding:14px; 
        margin-bottom:12px; 
        border-radius:10px; 
        background-color:#FFFFFF; 
        border:1px solid #E6E9EF;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <h3 style="color:#0A65FF; margin-bottom:4px;">{row['keyword'].capitalize()}</h3>
        <p style="font-size:15px; margin:0; color:#0A1A2F;">
            <strong>Frequency:</strong> {row['count']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SECTION 1 — BAR CHART
# =====================================================
st.subheader("🔑 Most Common Fraud Keywords")

bar_chart = (
    alt.Chart(keyword_freq.head(20))
    .mark_bar(color="#0A65FF")
    .encode(
        x=alt.X("count:Q", title="Frequency"),
        y=alt.Y("keyword:N", sort="-x", title="Keyword"),
        tooltip=["keyword", "count"]
    )
    .properties(height=480)
)

st.altair_chart(bar_chart, use_container_width=True)

# =====================================================
# SECTION 2 — INTERACTIVE NETWORK GRAPH
# =====================================================
st.subheader("🕸️ Interactive Keyword Network")

pairs = []
for kw_list in df["keywords"]:
    if len(kw_list) > 1:
        pairs.extend(itertools.combinations(kw_list, 2))

pair_counts = {}
for a, b in pairs:
    pair = tuple(sorted([a, b]))
    pair_counts[pair] = pair_counts.get(pair, 0) + 1

MIN_EDGE_WEIGHT = 6
filtered_pairs = {pair: c for pair, c in pair_counts.items() if c >= MIN_EDGE_WEIGHT}

if not filtered_pairs:
    st.warning("Not enough keyword connections to build a network.")
else:
    net = Network(height="650px", width="100%", bgcolor="#FFFFFF", font_color="#0A1A2F")
    net.barnes_hut()

    freq_map = dict(zip(keyword_freq["keyword"], keyword_freq["count"]))

    for kw, freq in freq_map.items():
        if freq > 2:
            net.add_node(
                kw,
                label=kw,
                size=min(freq * 2, 45),
                color="#0A65FF"
            )

    for (a, b), weight in filtered_pairs.items():
        net.add_edge(a, b, value=weight, title=f"Co-occurrences: {weight}")

    net.set_options("""
{
  "nodes": {
    "font": { "size": 22, "face": "arial", "color": "#0A1A2F" },
    "shape": "dot",
    "borderWidth": 1
  },
  "edges": {
    "width": 2,
    "color": { "color": "#0A65FF", "highlight": "#003EAA" },
    "smooth": { "enabled": true, "type": "continuous" }
  },
  "physics": {
    "enabled": true,
    "stabilization": { "iterations": 150 },
    "barnesHut": {
      "gravitationalConstant": -2500,
      "centralGravity": 0.10,
      "springLength": 240,
      "springConstant": 0.010,
      "avoidOverlap": 1
    }
  },
  "interaction": {
    "hover": true,
    "zoomView": true,
    "dragView": true,
    "dragNodes": true
  }
}
""")

    net.save_graph("keyword_network.html")
    HtmlFile = open("keyword_network.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=650, scrolling=True)

# =====================================================
# SECTION 3 — FULL KEYWORD TABLE WITH CUSTOM SEARCH BAR
# =====================================================
st.subheader("📚 Full Fraud Keyword List")

# --- CUSTOM SEARCH BAR STYLE (MATCHES YOUR IMAGE)
search_html = """
<input type="text" id="customSearch" placeholder="Enter a keyword" 
style="
    width: 100%;
    padding: 14px 18px;
    font-size: 16px;
    border-radius: 25px;
    border: 1px solid #C3C7CF;
    background-color: #F5F7FA;
    color: #0A1A2F;
    outline: none;
">
<script>
const streamlitInput = document.getElementsByName("customSearch")[0];
</script>
"""

search_value = st.text_input(
    "",
    placeholder="Try: 'mail theft', 'investment fraud', 'AI trading', 'identity theft'...",
)

table = keyword_freq.copy()
if search_value:
    table = table[table["keyword"].str.contains(search_value, case=False)]

st.dataframe(table, use_container_width=True)




