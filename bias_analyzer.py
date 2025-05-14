# Role and Objective
# This script creates a dynamic bias chart for news sources by scraping their articles,
# analyzing them for bias and reliability, and saving the results to a CSV file.
# It runs daily at 6:00 PM CDT to provide up-to-date bias and reliability scores for a news briefing service.

# Instructions
# - Scrape up to 20 articles per source daily at 6:00 PM CDT.
# - Use RSS feeds as a fallback if direct scraping fails.
# - Analyze articles for bias using sentiment analysis and ideological keywords.
# - Analyze articles for reliability using citations, sensationalism, and factual indicators.
# - Update scores daily and save to dynamic_bias_chart.csv.
# - Log all steps for transparency and debugging, including failures to a separate log file.
# - Persist through failures: retry scraping up to 5 times, and reanalyze if scores are ambiguous.

# Dependencies and NLTK Data Downloads
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('brown', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please run the following manually in a Python shell:")
    print("import nltk")
    print("nltk.download('punkt')")
    print("nltk.download('averaged_perceptron_tagger')")
    print("nltk.download('wordnet')")
    print("nltk.download('brown')")
    exit(1)

try:
    import requests
    from bs4 import BeautifulSoup
    from textblob import TextBlob
    import pandas as pd
    import re
    from datetime import datetime, timedelta
    import time
    import random
    import feedparser  # For RSS feed parsing
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install the required libraries using:")
    print("pip install requests beautifulsoup4 pandas lxml textblob feedparser")
    exit(1)

# Define absolute paths
BASE_PATH = r"C:\Users\jonat\Desktop\News"
CSV_PATH = f"{BASE_PATH}\\dynamic_bias_chart.csv"
LOG_PATH = f"{BASE_PATH}\\failed_sources.log"

# Define sources with their URLs and verified RSS feeds
sources = {
    "usa-facts": {"url": "https://usafacts.org", "rss": "https://usafacts.org/feed/", "bias": 0, "reliability": 5},
    "pew-research": {"url": "https://www.pewresearch.org", "rss": "https://www.pewresearch.org/feed/", "bias": 0, "reliability": 5},
    "ap": {"url": "https://apnews.com", "rss": "https://apnews.com/hub/ap-top-news?format=rss", "bias": 0, "reliability": 4},
    "reuters": {"url": "https://www.reuters.com", "rss": "https://www.reuters.com/arc/outboundfeeds/news-section/top-news/?outputType=xml", "bias": 0, "reliability": 4},
    "the-hill": {"url": "https://thehill.com", "rss": "https://thehill.com/feed/", "bias": 0, "reliability": 4},
    "cnn": {"url": "https://www.cnn.com", "rss": "http://rss.cnn.com/rss/cnn_topstories.rss", "bias": -1, "reliability": 3},
    "msnbc": {"url": "https://www.msnbc.com", "rss": "https://www.msnbc.com/feeds/latest", "bias": -2, "reliability": 2},
    "the-nation": {"url": "https://www.thenation.com", "rss": "https://www.thenation.com/feed/?post_type=article", "bias": -3, "reliability": 1},
    "huffpost": {"url": "https://www.huffpost.com", "rss": "https://www.huffpost.com/rss/index.xml", "bias": -2, "reliability": 2},
    "fox-news": {"url": "https://www.foxnews.com", "rss": "https://moxie.foxnews.com/google-publisher/latest.xml", "bias": 2, "reliability": 2},
    "the-daily-wire": {"url": "https://www.dailywire.com", "rss": "https://www.dailywire.com/feeds/rss.xml", "bias": 3, "reliability": 1},
    "breitbart": {"url": "https://www.breitbart.com", "rss": "http://feeds.feedburner.com/breitbart", "bias": 3, "reliability": 1},
    "new-york-times": {"url": "https://www.nytimes.com", "rss": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "bias": -1, "reliability": 4},
    "axios": {"url": "https://www.axios.com", "rss": "https://api.axios.com/feed/", "bias": -1, "reliability": 4},
    "the-atlantic": {"url": "https://www.theatlantic.com", "rss": "https://www.theatlantic.com/feed/all/", "bias": -2, "reliability": 3},
    "wall-street-journal": {"url": "https://www.wsj.com", "rss": "https://feeds.a.dj.com/rss/RSSWorldNews.xml", "bias": 1, "reliability": 4},
    "national-review": {"url": "https://www.nationalreview.com", "rss": "https://www.nationalreview.com/feed/", "bias": 2, "reliability": 2},
    "npr": {"url": "https://www.npr.org", "rss": "https://feeds.npr.org/1001/rss.xml", "bias": -1, "reliability": 4},
    "bbc": {"url": "https://www.bbc.com", "rss": "http://feeds.bbci.co.uk/news/rss.xml", "bias": 0, "reliability": 4},
    "al-jazeera": {"url": "https://www.aljazeera.com", "rss": "https://www.aljazeera.com/xml/rss/all.xml", "bias": -1, "reliability": 3}
}

# Log failed sources to a file
def log_failed_source(source, url, reason):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Failed to scrape {source} ({url}): {reason}\n")

# Scrape articles with enhanced retry logic, user-agent rotation, and RSS fallback
def scrape_articles(source, source_url, rss_url, num_articles=20, max_retries=5, timeout=120):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
    ]

    # Try direct scraping first
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive'
            }
            response = requests.get(source_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # Broaden the search for articles
            headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'])
            potential_articles = []
            for headline in headlines:
                parent = headline.find_parent(['article', 'div', 'section', 'li'])
                if parent and not any(cls in parent.get('class', []) for cls in ['nav', 'footer', 'sidebar', 'header']) and not any(id in parent.get('id', '') for id in ['header', 'footer', 'sidebar']):
                    potential_articles.append(parent)
            
            # Include links to articles
            links = soup.find_all('a', href=re.compile(r'/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+'))
            for link in links:
                parent = link.find_parent(['article', 'div', 'section', 'li'])
                if parent and parent not in potential_articles and not any(cls in parent.get('class', []) for cls in ['nav', 'footer', 'sidebar', 'header']):
                    potential_articles.append(parent)
            
            potential_articles = potential_articles[:50]  # Limit to 50 to reduce load
            
            print(f"Found {len(potential_articles)} potential articles at {source_url} on attempt {attempt + 1}.")
            if not potential_articles:
                print(f"No matching HTML elements found at {source_url} on attempt {attempt + 1}.")
                print(f"Sample HTML: {soup.prettify()[:500]}...")
                raise ValueError("No matching elements found")

            texts = [article.get_text(strip=True) for article in potential_articles if article.get_text(strip=True) and len(article.get_text(strip=True)) > 20]
            if not texts:
                print(f"No valid articles (length > 20 chars) found at {source_url} on attempt {attempt + 1}.")
                print(f"Found {len(potential_articles)} elements, but none met the length requirement.")
                raise ValueError("No articles extracted")
            
            print(f"Successfully scraped {len(texts)} articles from {source_url}.")
            time.sleep(random.uniform(1, 3))  # Random delay to avoid rate limits
            return texts[:num_articles]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {source_url}: {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached for {source_url}. Trying RSS feed...")
                break
            time.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter

    # Fallback to RSS feed
    try:
        print(f"Attempting to scrape RSS feed: {rss_url}")
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            log_failed_source(source, rss_url, "No entries in RSS feed")
            return []
        texts = []
        for entry in feed.entries[:num_articles]:
            content = entry.get('summary', entry.get('description', entry.get('title', '')))
            if content and len(content) > 20:
                texts.append(content)
        if not texts:
            log_failed_source(source, rss_url, "No valid articles in RSS feed")
            return []
        print(f"Successfully scraped {len(texts)} articles from RSS feed for {source}.")
        return texts
    except Exception as e:
        log_failed_source(source, rss_url, f"RSS feed failed: {e}")
        return []

# Calculate bias with enhanced chain-of-thought reasoning
def calculate_bias(articles, source):
    if not articles:
        print(f"No articles found for {source}. Defaulting to neutral bias (0).")
        return 0

    print(f"\nCalculating bias for {source}...")
    # Step 1: Summarize sentiment and themes
    sentiments = []
    for i, article in enumerate(articles, 1):
        blob = TextBlob(article)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
        print(f"Article {i}: Sentiment = {sentiment:.2f}, Sample: {article[:100]}...")
    avg_sentiment = sum(sentiments) / len(sentiments)
    print(f"Average sentiment: {avg_sentiment:.2f}")

    # Step 2: Identify ideological keywords
    left_keywords = ['social justice', 'systemic inequality', 'progressive', 'liberal', 'climate justice', 'equity', 'diversity', 'inclusion', 'critical race theory', 'universal healthcare', 'woke', 'reparations', 'defund', 'socialism', 'intersectionality', 'anti-racist', 'wealth tax']
    right_keywords = ['traditional values', 'free market', 'conservative', 'liberty', 'patriot', 'family values', 'capitalism', 'nationalism', 'second amendment', 'pro-life', 'globalist', 'deep state', 'maga', 'patriotism', 'border security']
    left_count, right_count = 0, 0
    for article in articles:
        article_lower = article.lower()
        left_count += sum(1 for keyword in left_keywords if keyword in article_lower)
        right_count += sum(1 for keyword in right_keywords if keyword in article_lower)
    print(f"Left-leaning keywords: {left_count}, Right-leaning keywords: {right_count}")

    # Step 3: Detect sarcasm or mixed sentiment
    sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments) if len(sentiments) > 1 else 0
    print(f"Sentiment variance: {sentiment_variance:.2f}")
    trimmed_sentiments = sentiments  # Default to full list
    if sentiment_variance > 0.5:  # High variance may indicate sarcasm or mixed sentiment
        print("High sentiment variance detected. Reanalyzing for consistency...")
        sentiments.sort()
        trimmed_sentiments = sentiments[1:-1] if len(sentiments) > 2 else sentiments
        avg_sentiment = sum(trimmed_sentiments) / len(trimmed_sentiments)
        print(f"Trimmed average sentiment (excluding outliers): {avg_sentiment:.2f}")

    # Step 4: Determine bias score
    bias_adjustment = 0
    if left_count > right_count + 3:  # Stronger adjustment for larger differences
        bias_adjustment = -3
    elif left_count > right_count + 1:
        bias_adjustment = -1
    elif right_count > left_count + 3:
        bias_adjustment = 3
    elif right_count > left_count + 1:
        bias_adjustment = 1
    print(f"Bias adjustment based on keywords: {bias_adjustment}")

    if avg_sentiment < -0.3 or (avg_sentiment < -0.15 and bias_adjustment < 0):
        bias = -3
    elif avg_sentiment < -0.15 or (avg_sentiment < -0.03 and bias_adjustment < 0):
        bias = -2
    elif avg_sentiment < -0.03 or bias_adjustment < 0:
        bias = -1
    elif avg_sentiment > 0.3 or (avg_sentiment > 0.15 and bias_adjustment > 0):
        bias = 3
    elif avg_sentiment > 0.15 or (avg_sentiment > 0.03 and bias_adjustment > 0):
        bias = 2
    elif avg_sentiment > 0.03 or bias_adjustment > 0:
        bias = 1
    else:
        bias = 0
    print(f"Initial bias score: {bias}")

    # Step 5: Reflect on the score
    historical_bias = sources[source]["bias"]
    if abs(bias - historical_bias) > 2 or (abs(bias - historical_bias) > 1 and left_count + right_count < 2):  # Adjust if keyword detection is low
        print(f"Warning: Bias score ({bias}) differs significantly from historical bias ({historical_bias}). Reanalyzing...")
        trimmed_avg = sum(trimmed_sentiments) / len(trimmed_sentiments)
        if abs(trimmed_avg - avg_sentiment) > 0.05:
            avg_sentiment = trimmed_avg
            bias = -2 if avg_sentiment < -0.15 else 2 if avg_sentiment > 0.15 else historical_bias
        else:
            bias = historical_bias if abs(bias - historical_bias) > 1 else bias
        print(f"Adjusted bias score: {bias}")
    print(f"Final bias score: {bias}")
    return bias

# Calculate reliability with enhanced chain-of-thought reasoning
def calculate_reliability(articles, source):
    if not articles:
        print(f"No articles found for {source}. Defaulting to average reliability (3).")
        return 3

    print(f"\nCalculating reliability for {source}...")
    # Step 1: Initialize base score
    score = 3
    print(f"Starting with base reliability score: {score}")

    # Step 2: Analyze each article for citations, sensationalism, and factual indicators
    for i, article in enumerate(articles, 1):
        article_lower = article.lower()
        has_citations = bool(re.search(r'\b(according to|reported by|source:|data from)\b', article_lower))
        has_sensationalism = bool(re.search(r'\b(shocking|outrageous|disastrous|scandal|you won\’t believe|shocking truth)\b', article_lower))
        has_opinionated_language = bool(re.search(r'\b(should|must|ought to|opinion|editorial)\b', article_lower))
        has_factual_indicators = bool(re.search(r'\b(20\d{2}|\d{1,2}%\s|\d{1,2}\spercent|\d{1,2}\s[a-zA-Z]+\s20\d{2})\b', article_lower))
        score_change = 0
        if has_citations:
            score_change += 0.7  # Increased weight to allow higher reliability for neutral sources
            print(f"Article {i}: Found citations, increasing score by 0.7")
        if has_factual_indicators:
            score_change += 0.05
            print(f"Article {i}: Found factual indicators (dates/percentages), increasing score by 0.05")
        if has_sensationalism:
            score_change -= 0.5
            print(f"Article {i}: Found sensational language, decreasing score by 0.5")
        if has_opinionated_language:
            score_change -= 0.2  # Reduced penalty to allow higher reliability
            print(f"Article {i}: Found opinionated language, decreasing score by 0.2")
        score += score_change

    # Step 3: Cap the score and apply fallback
    score = max(0, min(5, score))
    print(f"Score after analysis: {score}")
    if score < 3 and not any(re.search(r'\b(shocking|outrageous|disastrous|scandal|you won\’t believe|shocking truth)\b', article.lower()) for article in articles):
        score = max(3, score)
        print(f"Fallback applied: No sensationalism detected, ensuring reliability is at least 3")

    # Step 4: Reflect on the score
    historical_reliability = sources[source]["reliability"]
    if abs(score - historical_reliability) > 2:
        print(f"Warning: Reliability score ({score}) differs significantly from historical reliability ({historical_reliability}). Adjusting...")
        if score > historical_reliability:
            score = min(score, historical_reliability + 2)
        else:
            score = max(score, historical_reliability - 2)
        print(f"Adjusted reliability score: {score}")
    print(f"Final reliability score: {score}")
    return score

# Update the bias chart
def update_bias_chart():
    print(f"Updating bias chart at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    successful_sources = 0
    total_articles = 0
    failed_sources = 0

    for source, info in sources.items():
        try:
            print(f"\nProcessing {source}...")
            # Step 1: Scrape articles with retries and RSS fallback
            articles = scrape_articles(source, info["url"], info["rss"])
            if not articles:
                print(f"No articles scraped for {source}. Retaining previous scores: Bias = {info['bias']}, Reliability = {info['reliability']}")
                failed_sources += 1
                continue

            total_articles += len(articles)
            successful_sources += 1

            # Step 2: Analyze articles for bias
            bias = calculate_bias(articles, source)
            # Step 3: Analyze articles for reliability
            reliability = calculate_reliability(articles, source)
            # Step 4: Update scores
            sources[source]["bias"] = bias
            sources[source]["reliability"] = reliability
            sources[source]["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Updated {source}: Bias = {bias}, Reliability = {reliability}")
        except Exception as e:
            print(f"Error processing {source}: {e}")
            log_failed_source(source, info["url"], f"Unexpected error: {e}")
            print(f"Retaining previous scores for {source}: Bias = {info['bias']}, Reliability = {info['reliability']}")
            failed_sources += 1

    # Step 5: Save to CSV with explicit index_label
    try:
        df = pd.DataFrame.from_dict(sources, orient='index')
        print(f"DataFrame columns before saving: {df.columns.tolist()}")
        df.to_csv(CSV_PATH, index_label='index')
        print(f"Bias chart updated and saved to {CSV_PATH}")
    except Exception as e:
        print(f"Error saving bias chart to CSV: {e}")
        log_failed_source("CSV Save", CSV_PATH, f"Save error: {e}")

    # Step 6: Log summary
    print(f"\nSummary:")
    print(f"Total sources processed: {len(sources)}")
    print(f"Successful sources: {successful_sources}")
    print(f"Failed sources: {failed_sources}")
    print(f"Total articles scraped: {total_articles}")
    print("Bias chart update process completed.")

# Main execution
if __name__ == "__main__":
    update_bias_chart()
