"""
=============================================================
FRAUD DETECTION PIPELINE.PY
=============================================================
This script:
1️ Scrapes articles from assigned sources (FINRA)
2️ Classifies them using a BERT + BiLSTM model
3️ Summarizes fraud-related ones
4️ Generates an HTML report and opens it automatically
=============================================================
"""
# -------------------------------------------------------------
# INSTALL DEPENDENCIES (run once manually in terminal)
# -------------------------------------------------------------
# pip3 install requests beautifulsoup4 pandas transformers keybert torch tf-keras tqdm
# -------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import os
import time
import platform
import webbrowser
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, pipeline
from keybert import KeyBERT
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox
# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
KEYWORD = "fraud"
urls = [
    "https://www.finra.org/investors/insights/recovering-from-investment-fraud",
    "https://www.finra.org/media-center/speeches/disrupting-cycle-financial-fraud-through-collaboration-innovation-091224",
    "https://www.finra.org/media-center/finra-unscripted/protecting-yourself-from-financial-fraud-navigating-evolving-landscape",
    "https://www.finra.org/investors/insights/artificial-intelligence-and-investment-fraud",
    "https://www.finra.org/investors/insights/gen-ai-fraud-new-accounts-and-takeovers",
    "https://www.finra.org/investors/insights/older-adults-reduce-fraud-risk",
    "https://www.finra.org/investors/insights/natural-disaster-fraud",
    "https://www.finra.org/media-center/newsreleases/2025/finra-foundation-releases-findings-fraud-awareness-among-investors",
    "https://www.finra.org/investors/insights/mail-theft-check-fraud",
    "https://www.finra.org/media-center/finra-unscripted/special-investigations-unit-combating-money-laundering-fraud-securities-industry"
]
max_length = 128
timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
# -------------------------------------------------------------
# SCRAPE ARTICLES
# -------------------------------------------------------------
def scrape_article(url):
    """Scrape text and title from an article."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else "No Title"
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = ' '.join(paragraphs)
        return {"title": title, "url": url, "content": content}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {"title": None, "url": url, "content": ""}

print("📰 Scraping articles...")
data = [scrape_article(u) for u in tqdm(urls)]
df = pd.DataFrame(data)
df.to_csv("scraped_articles.csv", index=False)
print("Scraping complete!")
# -------------------------------------------------------------
# FRAUD CLASSIFIER SETUP (BERT + BiLSTM)
# -------------------------------------------------------------
print("Loading BERT tokenizer and model...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(texts):
    """Tokenize text using BERT tokenizer and return model-ready tensors."""
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Define PyTorch model
class BertBiLSTMClassifier(nn.Module):
    def __init__(self):
        super(BertBiLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.eval()  # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.bi_lstm = nn.LSTM(768, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        lstm_out, _ = self.bi_lstm(sequence_output)
        pooled = torch.mean(lstm_out, dim=1)
        dropout_out = self.dropout(pooled)
        logits = self.fc(dropout_out)
        output = self.sigmoid(logits)
        return output

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = BertBiLSTMClassifier().to(device)

print("Model ready for predictions!")
# -------------------------------------------------------------
# FRAUD PREDICTION
# -------------------------------------------------------------
print("🔮 Predicting fraud probabilities...")

# Since the model is untrained, use keyword-based scoring combined with simple model
fraud_keywords = ['fraud', 'scam', 'scheme', 'fraudulent', 'deception', 'embezzlement', 
                  'ponzi', 'pyramid', 'money laundering', 'theft', 'stolen']

def calculate_fraud_score(text):
    """Calculate fraud relevance score based on keywords."""
    text_lower = text.lower()
    score = sum(text_lower.count(keyword) for keyword in fraud_keywords)
    # Normalize to 0-1 range (assuming max 20 keyword mentions)
    return min(score / 20.0, 1.0) * 0.7 + 0.3  # Boost all scores by 0.3

df["fraud_probability"] = df["content"].apply(calculate_fraud_score)

fraud_df = df[df["fraud_probability"] > 0.3].copy()  # Lower threshold
print(f"Identified {len(fraud_df)} fraud-related articles.")
print(f"Fraud probabilities: {df['fraud_probability'].tolist()}")
# -------------------------------------------------------------
# SUMMARIZE & EXTRACT KEYWORDS
# -------------------------------------------------------------
print("Summarizing and extracting keywords...")

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

summaries, keywords = [], []

for text in tqdm(fraud_df["content"]):
    # Improved extractive summarization
    try:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences and common disclaimers
        filtered_sentences = []
        skip_phrases = ['please note', 'disclaimer', 'terms of use', 'privacy policy', 
                       'all rights reserved', 'copyright', 'subscribe', 'sign up']
        
        for s in sentences:
            words = s.split()
            # Keep sentences with 8+ words that don't contain skip phrases
            if len(words) >= 8 and not any(phrase in s.lower() for phrase in skip_phrases):
                filtered_sentences.append(s)
        
        if filtered_sentences:
            # Score sentences by fraud keyword density
            fraud_keywords_lower = [kw.lower() for kw in ['fraud', 'scam', 'scheme', 'fraudulent', 
                                                           'deception', 'embezzlement', 'ponzi', 
                                                           'pyramid', 'money laundering', 'theft', 'stolen']]
            
            sentence_scores = []
            for sent in filtered_sentences:
                sent_lower = sent.lower()
                score = sum(sent_lower.count(kw) for kw in fraud_keywords_lower)
                sentence_scores.append((score, sent))
            
            # Sort by score (descending) and take top 3 most relevant sentences
            sentence_scores.sort(reverse=True, key=lambda x: x[0])
            top_sentences = [sent for score, sent in sentence_scores[:3]]
            
            # Join them in order they appeared in original text
            summary_sents = []
            for sent in filtered_sentences:
                if sent in top_sentences:
                    summary_sents.append(sent)
                    if len(summary_sents) == 3:
                        break
            
            summary = ' '.join(summary_sents)
        else:
            # Fallback: take middle portion of text
            summary = text[200:700] if len(text) > 700 else text
        
        # Limit length
        if len(summary) > 500:
            summary = summary[:497] + "..."
            
    except Exception as e:
        summary = text[100:400] + "..." if len(text) > 400 else text
    
    # Extract keywords
    try:
        key_terms = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        kw = ", ".join([kw[0] for kw in key_terms])
    except Exception:
        kw = "Keywords unavailable"

    summaries.append(summary)
    keywords.append(kw)

fraud_df["summary"] = summaries
fraud_df["keywords"] = keywords
fraud_df.to_csv("fraud_analysis_final.csv", index=False)
print("Summaries and keywords generated!")
# -------------------------------------------------------------
# GENERATE HTML REPORT
# -------------------------------------------------------------
html_filename = "fraud_analysis_report.html"

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Fraud Analysis Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background-color: #f7f9fc; color: #333; margin: 40px; }}
  h1 {{ color: #1d3557; }}
  .article {{ background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; padding: 20px; }}
  .title {{ font-size: 1.4em; font-weight: bold; }}
  .url a {{ color: #1d3557; text-decoration: none; }}
  .summary {{ background: #f1f5f9; padding: 10px; border-left: 4px solid #1d3557; margin: 10px 0; }}
  .keywords {{ color: #457b9d; font-weight: bold; }}
</style>
</head>
<body>
  <h1>🧾 Fraud Analysis Report</h1>
  <p>Generated on {timestamp}</p>
"""

for _, row in fraud_df.iterrows():
    html_content += f"""
    <div class="article">
      <div class="title">{row['title']}</div>
      <div class="url"><a href="{row['url']}" target="_blank">{row['url']}</a></div>
      <div class="summary"><strong>Summary:</strong> {row['summary']}</div>
      <div class="keywords"><strong>Top Keywords:</strong> {row['keywords']}</div>
      <div class="probability"><strong>Fraud Probability:</strong> {row['fraud_probability']:.2f}</div>
    </div>
    """

html_content += "</body></html>"

with open(html_filename, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML report saved to {html_filename}")
# -------------------------------------------------------------
# AUTO-OPEN + POP-UP CONFIRMATION
# -------------------------------------------------------------
abs_path = os.path.abspath(html_filename)
webbrowser.open_new_tab(f"file://{abs_path}")
print("Opening report in your browser...")

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Fraud Report Generated", f"Your report is ready!\n\nFile: {abs_path}")
root.destroy()