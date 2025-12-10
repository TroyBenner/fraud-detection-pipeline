import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# ==========================================
# FINRA Index Pages to Pull From Automatically
# ==========================================

FINRA_SOURCES = [
    # Press Releases
    "https://www.finra.org/media-center/news-releases",

    # Enforcement Actions
    "https://www.finra.org/rules-guidance/enforcement/actions",

    # Disciplinary Actions
    "https://www.finra.org/rules-guidance/oversight-enforcement/disciplinary-actions",
]

# Fraud Keywords
FRAUD_KEYWORDS = [
    "fraud", "misconduct", "manipulation", "scheme", 
    "misleading", "securities fraud", "scam", "deceptive"
]


# ==========================================
# Helper Functions
# ==========================================

def is_fraud_related(text):
    """Return True if text matches any fraud keyword."""
    if not text:
        return False
    text = text.lower()
    return any(keyword in text for keyword in FRAUD_KEYWORDS)


def fetch_index_page(url):
    """Download index page HTML."""
    print(f"[+] Fetching index page: {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        print(f"[!] Failed to load index page ({response.status_code})")
        return None

    return BeautifulSoup(response.text, "html.parser")


def extract_article_links(soup):
    """Extract article URLs from index page."""
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.finra.org" + href

        # Must contain a FINRA article-style path
        if any(x in href for x in ["/media-center/", "/enforcement/", "/disciplinary-actions/"]):
            links.append(href)

    return list(set(links))  # unique


def scrape_article(url):
    """Scrape title, date, and content from an article URL."""
    print(f"[+] Scraping article: {url}")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"[!] Failed to load article ({response.status_code})")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title = title.get_text(strip=True) if title else "No Title"

        # Publication Date (REQUIRED)
        date_tag = soup.find("time")
        if not date_tag or not date_tag.has_attr("datetime"):
            print("[!] No publication date found — skipping.")
            return None

        article_date = date_tag["datetime"]

        # Body Text
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        # Filter out articles not related to fraud
        if not (is_fraud_related(title) or is_fraud_related(content)):
            print("[!] Not fraud-related — skipping.")
            return None

        return {
            "title": title,
            "url": url,
            "article_date": article_date,
            "content": content
        }

    except Exception as e:
        print(f"[ERROR] Exception scraping article: {e}")
        return None


# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    all_links = set()

    # Fetch all index pages
    for source in FINRA_SOURCES:
        soup = fetch_index_page(source)
        if soup:
            links = extract_article_links(soup)
            all_links.update(links)

        time.sleep(1)  # avoid hammering FINRA servers

    print(f"[✓] Total candidate articles found: {len(all_links)}")

    results = []

    # Scrape each article
    for url in all_links:
        article = scrape_article(url)
        if article:
            results.append(article)
        time.sleep(0.5)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("fraud_articles.csv", index=False, encoding="utf-8")

    print(f"[✓] Saved {len(results)} fraud articles to fraud_articles.csv")


if __name__ == "__main__":
    main()
