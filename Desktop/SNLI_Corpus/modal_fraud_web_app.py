"""
=============================================================
MODAL FRAUD DETECTION WEB APP
=============================================================
This creates a live web app that runs fraud detection and 
displays results at a public URL.
=============================================================
"""
import modal

# Create Modal app
app = modal.App("fraud-detector-web")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "requests",
        "beautifulsoup4",
        "pandas",
        "transformers",
        "keybert",
        "torch",
        "tqdm",
    )
)

# Web endpoint that serves the fraud report
@app.function(image=image, timeout=600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import re
    from datetime import datetime
    from keybert import KeyBERT
    
    web_app = FastAPI()
    
    @web_app.get("/", response_class=HTMLResponse)
    async def get_report():
        # URLs to scrape
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
        
        timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        
        # Scrape articles
        def scrape_article(url):
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string.strip() if soup.title else "No Title"
                paragraphs = [p.get_text() for p in soup.find_all('p')]
                content = ' '.join(paragraphs)
                return {"title": title, "url": url, "content": content}
            except Exception as e:
                return {"title": "Error", "url": url, "content": ""}
        
        data = [scrape_article(u) for u in urls]
        df = pd.DataFrame(data)
        
        # Calculate fraud scores
        fraud_keywords = ['fraud', 'scam', 'scheme', 'fraudulent', 'deception', 'embezzlement', 
                          'ponzi', 'pyramid', 'money laundering', 'theft', 'stolen']
        
        def calculate_fraud_score(text):
            text_lower = text.lower()
            score = sum(text_lower.count(keyword) for keyword in fraud_keywords)
            return min(score / 20.0, 1.0) * 0.7 + 0.3
        
        df["fraud_probability"] = df["content"].apply(calculate_fraud_score)
        fraud_df = df[df["fraud_probability"] > 0.3].copy()
        
        # Extract keywords
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        
        summaries, keywords = [], []
        
        for text in fraud_df["content"]:
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
                    
            except Exception:
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
        
        # Generate HTML with improved styling
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Analysis Report - Live</title>
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  
  body {{ 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #333;
    padding: 20px;
    min-height: 100vh;
  }}
  
  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}
  
  .header {{
    background: white;
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    text-align: center;
  }}
  
  h1 {{ 
    color: #1d3557;
    font-size: 2.5em;
    margin-bottom: 10px;
  }}
  
  .timestamp {{
    color: #666;
    font-size: 1.1em;
  }}
  
  .stats {{
    display: flex;
    gap: 20px;
    margin: 20px 0;
    justify-content: center;
    flex-wrap: wrap;
  }}
  
  .stat-card {{
    background: #f0f4ff;
    padding: 20px 30px;
    border-radius: 12px;
    text-align: center;
  }}
  
  .stat-number {{
    font-size: 2em;
    font-weight: bold;
    color: #667eea;
  }}
  
  .stat-label {{
    color: #666;
    margin-top: 5px;
  }}
  
  .article {{ 
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    padding: 30px;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  
  .article:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
  }}
  
  .title {{ 
    font-size: 1.5em;
    font-weight: 600;
    color: #1d3557;
    margin-bottom: 15px;
  }}
  
  .url {{
    margin-bottom: 15px;
  }}
  
  .url a {{ 
    color: #667eea;
    text-decoration: none;
    font-size: 0.9em;
  }}
  
  .url a:hover {{
    text-decoration: underline;
  }}
  
  .summary {{ 
    background: #f8fafc;
    padding: 20px;
    border-left: 4px solid #667eea;
    margin: 15px 0;
    border-radius: 8px;
    line-height: 1.6;
  }}
  
  .keywords {{ 
    color: #667eea;
    font-weight: 600;
    margin: 10px 0;
    padding: 10px;
    background: #f0f4ff;
    border-radius: 8px;
  }}
  
  .probability {{
    display: inline-block;
    background: #10b981;
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
  }}
  
  .footer {{
    text-align: center;
    color: white;
    margin-top: 40px;
    padding: 20px;
  }}
  
  .refresh-btn {{
    background: white;
    color: #667eea;
    border: none;
    padding: 12px 30px;
    border-radius: 25px;
    font-size: 1em;
    font-weight: 600;
    cursor: pointer;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
  }}
  
  .refresh-btn:hover {{
    transform: scale(1.05);
  }}
  
  @media (max-width: 768px) {{
    body {{
      padding: 10px;
    }}
    
    h1 {{
      font-size: 1.8em;
    }}
    
    .header {{
      padding: 20px;
    }}
    
    .article {{
      padding: 20px;
    }}
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔍 Fraud Detection Analysis</h1>
      <p class="timestamp">Generated on {timestamp}</p>
      
      <div class="stats">
        <div class="stat-card">
          <div class="stat-number">{len(df)}</div>
          <div class="stat-label">Articles Analyzed</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">{len(fraud_df)}</div>
          <div class="stat-label">Fraud Articles Found</div>
        </div>
      </div>
      
      <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Analysis</button>
    </div>
"""
        
        for _, row in fraud_df.iterrows():
            html_content += f"""
    <div class="article">
      <div class="title">{row['title']}</div>
      <div class="url"><a href="{row['url']}" target="_blank">🔗 {row['url']}</a></div>
      <div class="summary"><strong>📝 Summary:</strong><br>{row['summary']}</div>
      <div class="keywords"><strong>🏷️ Keywords:</strong> {row['keywords']}</div>
      <div><span class="probability">Fraud Score: {row['fraud_probability']:.0%}</span></div>
    </div>
"""
        
        html_content += """
    <div class="footer">
      <p>Powered by Modal + AI | Real-time Fraud Detection</p>
      <p style="margin-top: 10px; font-size: 0.9em;">Data sourced from FINRA</p>
    </div>
  </div>
</body>
</html>
"""
        
        return html_content
    
    return web_app
