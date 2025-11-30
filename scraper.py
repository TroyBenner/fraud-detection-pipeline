import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime
import spacy

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Transformers + KeyBERT
from transformers import pipeline
from keybert import KeyBERT

# Config
SEARCH_WORD = "fraud"
BASE_URL = "https://www.finra.org"
MAX_LINKS = 30

ALLOWED_SOURCES = [
    "https://www.finra.org/media-center/news-releases/",
    "https://www.finra.org/investors/insights/",
    "https://www.finra.org/media-center/",
]

CSV_OUTPUT_PATH = "./articles-fraud.csv"

#nlp model
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn"
)

#clean text
def clean_text(text: str):
    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(tokens)

#extract text from articles
def extract_text(url: str):
    try:
        r = requests.get(url, timeout=10)
        soup = bs(r.text, "html.parser")

        # FINRA normally stores content here
        div = soup.find("div", class_="block-region-middle")

        if not div:
            # fallback to body
            return soup.get_text(" ", strip=True)

        return div.get_text(" ", strip=True)

    except Exception:
        return ""

# -------------------------
# EXTRACT TITLE
# -------------------------
def extract_title(url: str):
    try:
        r = requests.get(url, timeout=10)
        soup = bs(r.text, "html.parser")

        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return soup.title.get_text(strip=True) if soup.title else "Unknown Title"
    except:
        return "Unknown Title"

#scrape list links
def is_allowed(url):
    return any(url.startswith(prefix) for prefix in ALLOWED_SOURCES)


def get_article_links(keyword: str, max_links: int = 20):
    driver = webdriver.Chrome()
    driver.maximize_window()
    wait = WebDriverWait(driver, 10)

    driver.get(BASE_URL)
    search_input = wait.until(
        EC.visibility_of_element_located((By.CLASS_NAME, "custom-landing-search"))
    )
    search_input.send_keys(keyword + Keys.ENTER)

    urls = set()

    while len(urls) < max_links:
        wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "search-url")))
        results = driver.find_elements(By.CLASS_NAME, "search-url")

        for r in results:
            url = r.text.strip()
            if url and is_allowed(url):
                urls.add(url)
                if len(urls) >= max_links:
                    break

        # Try next page
        try:
            next_btn = driver.find_element(By.CLASS_NAME, "enabled")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(1)
        except:
            break

    driver.quit()
    return list(urls)

#nlp process
def summarize_text(text: str) -> str:
    try:
        result = summarizer(text[:3000], max_length=150, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except:
        return text[:300]


def extract_keywords(text: str):
    try:
        keywords = kw_model.extract_keywords(text, top_n=10)
        return ", ".join([kw for kw, score in keywords])
    except:
        return ""

#main
def main():
    print(f"Scraping up to {MAX_LINKS} FINRA articles…")
    links = get_article_links(SEARCH_WORD, max_links=MAX_LINKS)
    print(f"Found {len(links)} valid FINRA links.")

    rows = []

    for i, url in enumerate(links):
        print(f"[{i+1}/{len(links)}] Processing: {url}")

        raw_text = extract_text(url)
        if not raw_text or len(raw_text) < 200:
            print("Skipped (too short)")
            continue

        title = extract_title(url)
        cleaned = clean_text(raw_text)
        summary = summarize_text(raw_text)
        keywords = extract_keywords(cleaned)

        rows.append({
            "url": url,
            "title": title,
            "summary": summary,
            "keywords": keywords,
            "timestamp": datetime.now().isoformat()
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"CSV saved to: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
