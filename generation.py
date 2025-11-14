# ============================================================
# FINRA Fraud Analysis — Use Provided URLs (Playwright + BART)
# ============================================================
import time, random, platform
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from transformers import pipeline
from keybert import KeyBERT

# ---------- CONFIG ----------
URLS = [
    "https://www.finra.org/investors/insights/recovering-from-investment-fraud",
    "https://www.finra.org/media-center/speeches/disrupting-cycle-financial-fraud-through-collaboration-innovation-091224",
    "https://www.finra.org/media-center/finra-unscripted/protecting-yourself-from-financial-fraud-navigating-evolving-landscape",
    "https://www.finra.org/investors/insights/artificial-intelligence-and-investment-fraud",
    "https://www.finra.org/investors/insights/gen-ai-fraud-new-accounts-and-takeovers",
    "https://www.finra.org/investors/insights/older-adults-reduce-fraud-risk",
    "https://www.finra.org/investors/insights/natural-disaster-fraud",
    "https://www.finra.org/media-center/newsreleases/2025/finra-foundation-releases-findings-fraud-awareness-among-investors",
    "https://www.finra.org/investors/insights/mail-theft-check-fraud",
    "https://www.finra.org/media-center/finra-unscripted/special-investigations-unit-combating-money-laundering-fraud-securities-industry",
]
TS = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- SCRAPE (Playwright) ----------
SELECTORS = [
    "article",               # most pages
    "div.page-content",      # investor insights
    "div.field--name-body",  # drupal body
    "main"                   # fallback
]

def scrape_one(page, url, tries=2):
    for attempt in range(1, tries+1):
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            # wait for any of our containers
            found = False
            for sel in SELECTORS:
                try:
                    page.wait_for_selector(sel, timeout=8000)
                    found = True
                    content = page.inner_text(sel).strip()
                    if len(content) > 200:
                        break
                except PWTimeout:
                    continue
            if not found:
                # last resort: whole body text
                content = page.inner_text("body").strip()

            # title
            try:
                h1 = page.locator("h1").first
                title = (h1.inner_text().strip()) if h1.count() > 0 else page.title()
            except Exception:
                title = page.title()

            # de-noise obvious boilerplate (kept light)
            bad_bits = [
                "Use the top and side menus for direct access",
                "FINRA Data provides non-commercial use of data",
                "FINRA GATEWAY", "DR PORTAL", "FINPRO"
            ]
            for b in bad_bits:
                content = content.replace(b, " ")

            if len(content) < 200 and attempt < tries:
                # try again once (network hiccup / JS late load)
                time.sleep(1.0)
                continue

            return {"title": title, "url": url, "content": content}
        except Exception as e:
            if attempt >= tries:
                return {"title": f"Error: {url}", "url": url, "content": ""}

def scrape_all(urls):
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/124 Safari/537.36")
        page = ctx.new_page()
        for u in tqdm(urls, desc="Scraping articles", colour="cyan"):
            rows.append(scrape_one(page, u))
        browser.close()
    return pd.DataFrame(rows)

print(f"📰 Scraping {len(URLS)} provided FINRA URLs…")
t0 = time.time()
raw_df = scrape_all(URLS)
print(f"✅ Scraped in {(time.time()-t0)/60:.2f} min")

# Save raw for debugging/traceability
raw_df.to_csv("fraud_articles_raw.csv", index=False, encoding="utf-8")

# ---------- FILTER empty/404-like ----------
def looks_like_404(txt: str) -> bool:
    if not txt or len(txt.strip()) < 200:
        return True
    low = txt.lower()
    return any(k in low for k in ["page not found", "404", "use the top and side menus"])

work_df = raw_df[~raw_df["content"].apply(looks_like_404)].reset_index(drop=True)
if work_df.empty:
    print("⚠️ No usable article bodies were extracted. (If this persists, re-run; some pages can be slow to render.)")
    # still save empty final file with headers
    pd.DataFrame(columns=["title","url","summary","keywords","timestamp"]).to_csv(
        "fraud_analysis_final.csv", index=False, encoding="utf-8"
    )
    raise SystemExit(0)

# ---------- SUMMARIZE + KEYWORDS ----------
print("🧠 Summarizing + extracting keywords…")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

summaries, keywords = [], []
for txt in tqdm(work_df["content"], desc="Summarizing", colour="green"):
    body = txt[:4096]  # plenty for a high-quality summary
    try:
        s = summarizer(body, max_length=140, min_length=40, do_sample=False)[0]["summary_text"]
    except Exception:
        s = body[:300]
    try:
        terms = kw_model.extract_keywords(body, keyphrase_ngram_range=(1,2), stop_words="english", top_n=6)
        k = ", ".join([w for w,_ in terms])
    except Exception:
        k = ""
    summaries.append(s)
    keywords.append(k)

final_df = work_df.copy()
final_df["summary"] = summaries
final_df["keywords"] = keywords
final_df["timestamp"] = TS

# clean & save
final_df = (final_df
            .drop_duplicates(subset=["title","summary"])
            .loc[final_df["summary"].str.len() > 30, ["title","url","summary","keywords","timestamp"]]
            .reset_index(drop=True))

final_df.to_csv("fraud_analysis_final.csv", index=False, encoding="utf-8")
print(f"✅ Saved fraud_analysis_final.csv with {len(final_df)} articles.")

# ---------- Notify ----------
def celebrate(n):
    try:
        if platform.system() == "Windows":
            import winsound; winsound.MessageBeep(winsound.MB_ICONASTERISK)
        from plyer import notification
        notification.notify(
            title="✅ FINRA Fraud Analysis Complete",
            message=f"{n} articles processed and saved.",
            app_name="Fraud Analyzer",
            timeout=8
        )
    except Exception as e:
        print(f"(Notification skipped: {e})")

celebrate(len(final_df))
print("🎯 Done.")
