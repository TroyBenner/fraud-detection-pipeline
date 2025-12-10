import streamlit as st

def inject_light_ui():
    css = """
    <style>

    /* -----------------------------------------
       MAIN APP BACKGROUND
    ------------------------------------------*/
    .stApp {
        background-color: #FFFFFF !important;
    }

    /* -----------------------------------------
       SIDEBAR STYLING
    ------------------------------------------*/
    section[data-testid="stSidebar"] {
        background-color: #F5F7FA !important;
        border-right: 1px solid #E6E9EF !important;
        padding-top: 10px !important;
    }

    /* Force all sidebar text to be visible */
    section[data-testid="stSidebar"] * {
        color: #0A1A2F !important;
        font-weight: 500 !important;
    }

    /* -----------------------------------------
       CARD COMPONENTS
    ------------------------------------------*/
    .card {
        background-color: #FFFFFF !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        border: 1px solid #E6E9EF !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }

    /* -----------------------------------------
       HEADERS
    ------------------------------------------*/
    h1, h2, h3, h4 {
        color: #0A1A2F !important;
        font-weight: 700 !important;
    }

    /* -----------------------------------------
       PARAGRAPH TEXT
    ------------------------------------------*/
    p {
        color: #0A1A2F !important;
        font-size: 16px !important;
    }

    /* -----------------------------------------
       LINKS IN LIGHT MODE
    ------------------------------------------*/
    a {
        color: #0A65FF !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }

    a:hover {
        text-decoration: underline !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def sidebar_logo():
    st.sidebar.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <img src="https://i.imgur.com/lAVJ7Vx.png" 
                 style="width:70%; border-radius:12px;"/>

            <h2 style="color:#0A1A2F; margin-top:10px;">
                IntelliFraud
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
