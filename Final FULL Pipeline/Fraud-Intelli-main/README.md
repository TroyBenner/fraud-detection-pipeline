# 💠 FINRA Fraud Intelligence Engine

The **FINRA Fraud Intelligence Engine** is a sleek, modern Streamlit web app that uses BERT-based semantic search and transformer-generated explanations to help users explore investment fraud insights. It loads FINRA article summaries, finds the most relevant match for any question, explains articles in simple terms, and even allows comparison between articles.

---

## 🌟 Features

- **Semantic Search (BERT embeddings)** — returns the single most relevant FINRA article based on user input  
- **AI-Powered Article Explanations** — uses FLAN-T5 to rewrite summaries in simple terms  
- **Automatic Fraud Categorization** — tags articles by fraud type (AI Fraud, Check Fraud, Elder Fraud, Scams, etc.)  
- **Article Comparison Mode** — compare your query against another article  
- **Search History Sidebar**  
- **Clean, premium UI** with glassmorphism, animation, and modern styling  
- Fully deployable on **Streamlit Cloud** (free)

---

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py


