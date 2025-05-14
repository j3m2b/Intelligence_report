import pandas as pd
import requests
from datetime import datetime, timedelta
import textwrap
import re
from bs4 import BeautifulSoup
import feedparser

# Configuration
NEWS_API_KEY = "f3e5a71b5a64486b9b35741881cd6b67"  # Your API key
BASE_PATH = r"C:\Users\jonat\Desktop\News"
CSV_PATH = f"{BASE_PATH}\\dynamic_bias_chart.csv"
OUTPUT_FILE = f"{BASE_PATH}\\news_briefing.txt"
MAX_SUMMARY_LENGTH = 100  # Maximum words for article summaries

# Step 1: Load the dynamic bias chart
def load_bias_chart():
    try:
        df = pd.read_csv(CSV_PATH)
        required_columns = ['index', 'bias', 'reliability', 'url', 'rss']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Bias chart CSV is missing required columns: {missing_columns}. Found columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found. Please ensure the file exists.")
        exit(1)
    except Exception as e:
        print(f"Error loading bias chart: {e}")
        exit(1)

# Step 2: Fetch recent articles from NewsAPI
def fetch_articles_newsapi(source, from_date, to_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "sources": source,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "pageSize": 5  # Limit to 5 articles per source
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            print(f"No articles found for {source} in NewsAPI for the specified date range.")
        else:
            print(f"Fetched {len(articles)} articles for {source} via NewsAPI.")
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles for {source} via NewsAPI: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error while fetching articles for {source} via NewsAPI: {e}")
        return []

# Fallback: Fetch articles directly from RSS feed if NewsAPI fails
def fetch_articles_rss(rss_url, num_articles=5):
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            print(f"No entries found in RSS feed: {rss_url}")
            return []
        articles = []
        for entry in feed.entries[:num_articles]:
            article = {
                "title": entry.get('title', 'No title available'),
                "description": entry.get('summary', entry.get('description', '')),
                "content": entry.get('summary', entry.get('description', '')),
                "published": entry.get('published', datetime.now().strftime('%Y-%m-%d'))
            }
            articles.append(article)
        print(f"Fetched {len(articles)} articles via RSS for {rss_url}.")
        return articles
    except Exception as e:
        print(f"Error fetching articles via RSS for {rss_url}: {e}")
        return []

# Step 3: Summarize articles (take first MAX_SUMMARY_LENGTH words from description or content)
def summarize_article(article):
    description = article.get("description", "") or article.get("content", "") or article.get("title", "")
    if not description:
        print(f"Article skipped - no description, content, or title: {article}")
        return None  # Return None to skip articles with no content
    # Remove any HTML tags or unwanted characters
    description = re.sub(r'<[^>]+>', '', description)
    description = re.sub(r'\s+', ' ', description).strip()
    words = description.split()[:MAX_SUMMARY_LENGTH]
    if not words:
        print(f"Article skipped - empty after processing: {article}")
        return None  # Return None to skip empty articles
    summary = " ".join(words)
    return summary + "..." if len(words) == MAX_SUMMARY_LENGTH else summary

# Step 4: Generate the briefing
def generate_briefing():
    # Load bias chart
    bias_chart = load_bias_chart()
    
    # Define date range (adjusted for simulation - last 24 hours within NewsAPI's 30-day limit)
    to_date = "2024-10-14"  # Most recent date available
    from_date = "2024-10-13"
    
    # Unbiased Briefing: Sources with bias -0.5 to 0.5 and reliability >= 3.5
    unbiased_sources = bias_chart[(bias_chart['bias'].between(-0.5, 0.5)) & (bias_chart['reliability'] >= 3.5)]
    print(f"Unbiased Sources: {unbiased_sources['index'].tolist()}")
    
    # Leaning Perspectives: Sources with bias <= -1 or >= 1 and reliability <= 3
    leaning_sources = bias_chart[((bias_chart['bias'] <= -1) | (bias_chart['bias'] >= 1)) & (bias_chart['reliability'] <= 3)]
    print(f"Leaning Sources: {leaning_sources['index'].tolist()}")
    
    # Fetch and summarize articles for unbiased briefing
    unbiased_briefing = []
    for _, row in unbiased_sources.iterrows():
        source = row['index']
        articles = fetch_articles_newsapi(source, from_date, to_date)
        if not articles:  # Fallback to RSS if NewsAPI fails
            articles = fetch_articles_rss(row['rss'])
        for article in articles:
            summary = summarize_article(article)
            if summary is None:
                continue
            entry = f"{source.upper()}: {article['title']}\n{textwrap.fill(summary, width=80)}\n"
            unbiased_briefing.append(entry)
    
    # Fetch and summarize articles for leaning perspectives
    left_leaning = []
    right_leaning = []
    for _, row in leaning_sources.iterrows():
        source = row['index']
        bias = row['bias']
        articles = fetch_articles_newsapi(source, from_date, to_date)
        if not articles:  # Fallback to RSS if NewsAPI fails
            articles = fetch_articles_rss(row['rss'])
        for article in articles:
            summary = summarize_article(article)
            if summary is None:
                continue
            entry = f"{source.upper()}: {article['title']}\n{textwrap.fill(summary, width=80)}\n"
            if bias <= -1:
                left_leaning.append(entry)
            else:
                right_leaning.append(entry)
    
    # Write the briefing to a file
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            # Header
            f.write(f"News Briefing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Unbiased Briefing Section
            f.write("=== CIA-Style Unbiased Briefing ===\n\n")
            if unbiased_briefing:
                f.write("\n".join(unbiased_briefing))
            else:
                f.write("No articles available for unbiased briefing.\n")
            
            # Leaning Perspectives Section
            f.write("\n=== Leaning Perspectives ===\n")
            
            f.write("\nLeft-Leaning:\n")
            if left_leaning:
                f.write("\n".join(left_leaning))
            else:
                f.write("No left-leaning articles available.\n")
            
            f.write("\nRight-Leaning:\n")
            if right_leaning:
                f.write("\n".join(right_leaning))
            else:
                f.write("No right-leaning articles available.\n")
        
        print(f"Briefing generated and saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing briefing to file: {e}")

# Main execution
if __name__ == "__main__":
    generate_briefing()
