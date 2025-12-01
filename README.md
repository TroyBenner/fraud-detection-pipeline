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


## ⚡ Quick Start

### **1. Install dependencies (uv recommended)**

```bash
uv venv
uv pip install -r requirements.txt
```

Or using pip:

```bash
pip install -r requirements.txt
```

### **2. Environment Setup**

Copy the example environment file:


### **3. ⚙️ Running the application

```bash
pip install -r requirements.txt
streamlit run app.py
```
---

# 🖼️ Visual Overview


## 📸 Screenshots (Placeholders)

| Feature            | Screenshot              |
| ------------------ | ----------------------- |
| Home Search        | ![](images/home.png)    |
| Simple Explanation | ![](images/explain.png) |
| Comparison Mode    | ![](images/compare.png) |

---


# 📚 Data Source

All FINRA summaries come from the ** [FINRA]('https://www.finra.org/)** pages.
Data was collected using **web scraping (BeautifulSoup)**.

Each article entry looks like:

```json
{
  "title": "Avoiding Elder Financial Exploitation",
  "summary": "FINRA warns about schemes targeting older adults...",
  "url": "https://www.finra.org/investors/alerts/elder-fraud",
  "fraud_type": "Elder Fraud",
  "embedding": [ ... ]
}
```
---

## **Explainability (FLAN-T5)**

FLAN-T5 rewrites FINRA summaries in simple language:

* Converts financial jargon → plain English
* Focuses on “what happened” and “how to protect yourself”
* Makes alerts easier for general readers

---

## **Fraud Categorization**

Articles are automatically tagged with categories like:

* Elder Fraud
* Check Fraud
* AI Investment Scams
* Crypto Fraud
* Social Engineering
* Pump-and-Dump

Tagging is based on keyword clustering + semantic similarity.

---

## **Comparison Mode**

You can compare:

* Your question
  **against**
* Any FINRA article

The system embeds both and explains whether they are similar fraud patterns.

---

# 🔎 Example Queries

Try these in the app:

* **“How do elder fraud scams work?”**
* **“What are signs of crypto investment fraud?”**
* **“Compare AI trading scams with pump-and-dump schemes.”**
* **“What should older investors watch out for in phone scams?”**

---

# 📈 Findings & Why This Project Is Useful

| Benefit                   | Description                                         |
| ------------------------- | --------------------------------------------------- |
| **Semantic Search**       | Queries based on meaning, not keywords.             |
| **Simple Explanations**   | FLAN-T5 rewrites FINRA alerts in everyday language. |
| **Fraud Comparison**      | Understand how different scams relate.              |
| **Automatic Fraud Types** | Learn which fraud category the alert belongs to.    |
| **Clean UI**              | Easy for beginners or investors doing quick checks. |

---

# 📄 License / Usage Notes

FINRA content is public.
Only **summaries + URLs** are stored.
Full article bodies are **not** scraped or reproduced.

---

# 🤖 Technologies Used

* BERT (embeddings)
* FLAN-T5 (language simplification)
* Streamlit (UI)
* NumPy (vector search)
* BeautifulSoup (scraping)
* Python 3.10+



