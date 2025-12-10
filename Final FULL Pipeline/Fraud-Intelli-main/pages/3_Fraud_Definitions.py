import streamlit as st
import pandas as pd

# Theme UI (no sidebar logo)
from intellifraud_ui import inject_light_ui

# Supabase loader
from load_data_supabase import load_fraud_data

# Import your full dictionary of keyword definitions
from definitions import TERM_DEFINITIONS


# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Fraud Definitions", layout="wide")
inject_light_ui()


# ---------------------------------------------
# HERO HEADER
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
        📘 Fraud Glossary & Definitions
    </h1>
    <p style="font-size:17px; color:#0A1A2F;">
        Explore major fraud categories and a searchable glossary of key industry, regulatory, and cybersecurity terms.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------
# LOAD SUPABASE DATA
# ---------------------------------------------
@st.cache_data
def load_data():
    df = load_fraud_data()
    df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])
    return df

df = load_data()

all_keywords = sorted({kw.lower() for kw_list in df["keywords"] for kw in kw_list})


# ---------------------------------------------
# FRAUD CATEGORY DEFINITIONS
# ---------------------------------------------
FRAUD_TYPES = {
    "Investment Fraud": "Deceptive practices designed to mislead investors through false information, unrealistic promises, or manipulated financial data. These schemes often target individuals seeking high returns and may involve Ponzi structures, fake securities, or unregistered offerings.",
    "Securities Fraud": "Illegal activities involving misinformation, manipulation, or deception in financial markets. Examples include insider trading, falsified disclosures, pump-and-dump schemes, and market manipulation intended to distort asset values.",
    "Identity Theft": "The unauthorized use of someone’s personal information, such as Social Security numbers or financial credentials, typically used to open accounts, commit fraud, or steal funds. Identity theft is a major driver of downstream financial crimes.",
    "Account Takeover": "A cyber-enabled crime where attackers gain unauthorized access to a user’s financial or digital accounts. This may occur through phishing, credential theft, SIM swapping, or malware, allowing criminals to move money or alter security settings.",
    "Money Laundering": "The process of disguising illegally obtained funds to make them appear legitimate. Criminals use layering, structuring, and integration methods to hide transaction origins across accounts, companies, and jurisdictions.",
    "Wire Fraud": "A federal crime involving deception conducted via communications technology such as email, phone, or online messaging. Wire fraud schemes often involve impersonation, false claims, or manipulated payment instructions.",
    "Phishing": "A social-engineering technique in which attackers impersonate trusted entities to steal credentials, financial information, or personal data. Phishing is a leading cause of account compromise and financial crime.",
    "Elder Fraud": "Financial exploitation specifically targeting older adults through coercion, impersonation, or deceptive investment schemes. Elder fraud commonly involves romance scams, tech-support fraud, or high-pressure solicitation tactics.",
    "Affinity Fraud": "Scams targeting members of identifiable groups—such as religious, professional, or ethnic communities—where trust within the group is exploited to promote fraudulent investments or donations.",
    "Insider Trading": "Illegal trading in securities based on material non-public information obtained through professional or privileged access. Insider trading undermines market fairness and is subject to strict regulatory enforcement.",
    "Market Manipulation": "Intentional actions designed to distort the natural supply, demand, or pricing of financial instruments. Examples include wash trading, spoofing, pump-and-dump schemes, and coordinated misinformation campaigns.",
    "Ponzi Scheme": "A fraudulent investment structure where returns are paid to existing investors using money from new investors, rather than legitimate profit. Ponzi schemes collapse when recruitment slows and payouts exceed inflows.",
    "Pyramid Scheme": "A recruitment-based scam where participants earn compensation primarily through enrolling others rather than selling real products or services. These schemes are unsustainable and illegal in most jurisdictions.",
    "Embezzlement": "The theft or misappropriation of funds entrusted to an employee, agent, or fiduciary. Embezzlement involves abusing a position of trust and often includes unauthorized transfers or personal use of corporate assets.",
    "Cyber Fraud": "Fraud conducted through digital systems such as email, websites, mobile apps, or compromised networks. Cyber fraud includes malware-based theft, credential harvesting, ransomware, and business email compromise.",
    "Credit Card Fraud": "Unauthorized use of someone’s credit card information to make purchases or withdraw funds. This includes card-not-present fraud, skimming, synthetic identities, and compromised merchant systems.",
    "Mortgage Fraud": "Misrepresentation or falsification of information during a mortgage application or underwriting process. Examples include inflated income statements, false occupancy claims, and manipulated property valuations.",
    "Insurance Fraud": "Deceptive practices used to obtain insurance payouts not deserved. Common examples include staged accidents, exaggerated claims, false injuries, and falsified documentation.",
}


# ---------------------------------------------
# DISPLAY: FRAUD CATEGORIES
# ---------------------------------------------
st.subheader("🧭 Major Fraud Categories")

for fraud_type, definition in FRAUD_TYPES.items():
    st.markdown(f"""
    <div style="
        padding:14px; 
        margin-bottom:12px; 
        border-radius:10px; 
        background-color:#FFFFFF; 
        border:1px solid #E6E9EF;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <h3 style="color:#0A65FF; margin-bottom:6px;">{fraud_type}</h3>
        <p style="font-size:15px; color:#0A1A2F; line-height:1.6;">
            {definition}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------
# SEARCH BAR FOR KEYWORD GLOSSARY
# ---------------------------------------------
st.subheader("🔍 Search Glossary Terms")

search = st.text_input(
    "Search for a term:",
    placeholder="Type a keyword such as 'finra', 'markets', 'phishing', 'mobile'..."
).lower().strip()

filtered_keywords = (
    [kw for kw in all_keywords if search in kw] if search else all_keywords
)


# ---------------------------------------------
# DISPLAY: KEYWORD DEFINITIONS
# ---------------------------------------------
st.subheader("📖 Keyword Glossary from Articles")

if not filtered_keywords:
    st.info("No matching terms found.")
else:
    for term in filtered_keywords:
        definition = TERM_DEFINITIONS.get(term, "Definition not available.")
        st.markdown(f"""
        <div style="
            padding:14px; 
            margin-bottom:12px; 
            border-radius:10px; 
            background-color:#FFFFFF; 
            border:1px solid #E6E9EF;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);
        ">
            <h3 style="color:#0A65FF; margin-bottom:6px;">{term.capitalize()}</h3>
            <p style="font-size:15px; color:#0A1A2F; line-height:1.6;">
                {definition}
            </p>
        </div>
        """, unsafe_allow_html=True)
