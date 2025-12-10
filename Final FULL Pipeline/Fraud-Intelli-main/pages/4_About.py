import streamlit as st
from intellifraud_ui import inject_light_ui, sidebar_logo

# ------------------------------------------------------
# PAGE CONFIG & UI
# ------------------------------------------------------
st.set_page_config(page_title="About IntelliFraud", layout="wide")
inject_light_ui()

# ------------------------------------------------------
# LOGO AT THE TOP
# ------------------------------------------------------
st.markdown(
    """
<div style="text-align:center; margin-top:10px; margin-bottom:20px;">
    <img src="https://i.imgur.com/lAVJ7Vx.png" style="width:200px;">
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------
# HEADER HERO
# ------------------------------------------------------
st.markdown(
    """
<div style="
    padding: 25px;
    background: #F5F7FA;
    border-radius: 15px;
    border: 1px solid #E6E9EF;
    margin-bottom: 20px;
    text-align:center;
">
    <h1 style="color:#0A1A2F; margin-bottom:5px;">ℹ️ About IntelliFraud</h1>
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------
# ABOUT SECTION — CLEAN, NO RAW HTML VISIBLE
# ------------------------------------------------------
st.markdown(
    """
<div style="
    background-color:#FFFFFF;
    padding:25px;
    border-radius:12px;
    border:1px solid #E6E9EF;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
    line-height:1.6;
">

<p style="font-size:18px; color:#0A1A2F;">
<strong style="color:#0A65FF;">IntelliFraud</strong> is an interactive fraud-intelligence platform built to surface emerging
fraud patterns, regulatory enforcement activity, keyword signals, and risk indicators. It provides
a streamlined, data-driven environment for analysts, researchers, and stakeholders to explore trends
across financial and cyber fraud.
</p>

<p style="font-size:18px; color:#0A1A2F;">
The platform is powered by publicly available sources, including FINRA enforcement data, and uses
modern data science workflows built with <strong>Python</strong>, <strong>Streamlit</strong>, and <strong>Supabase</strong>.
IntelliFraud supports live search, pattern detection, keyword analytics, and automated insights.
</p>

<p style="font-size:18px; color:#0A1A2F;">
IntelliFraud was created by 
<strong>Clara Belluci</strong>,
<strong>Troy Benner</strong>,
<strong>Hoang Bui</strong>,
<strong>Tori-Ana McNeil</strong>,
in collaboration with the
<strong>UNC Charlotte School of Data Science</strong> and <strong>USAA</strong>.  
The project demonstrates real-world applications of data science, AI-driven investigation, and responsible financial analysis.
</p>

</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown(
    """
<div style="text-align:center; margin-top:40px; color:#666; font-size:14px;">
    This project is intended for educational and analytical purposes only.
</div>
""",
    unsafe_allow_html=True
)


