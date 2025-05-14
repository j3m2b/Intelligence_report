# run_intel_pipeline.py – full bias/sentiment analytics with dedupe, overlays, alerts, and drill-downs

import subprocess, os, re, base64
from datetime import datetime
from collections import Counter
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
stop_words = set(nltk.corpus.stopwords.words('english')).union(ENGLISH_STOP_WORDS)

# Paths
BASE_PATH = r"C:\Users\jonat\Desktop\News"
BRIEFING_SCRIPT = os.path.join(BASE_PATH, "briefing_generator.py")
HISTORY_LOG = os.path.join(BASE_PATH, "briefing_history.csv")
OUTPUT_HTML = os.path.join(BASE_PATH, "intelligence_report.html")

# Step 1: Run Briefing Generator
print("[INFO] Running Briefing Generator...")
briefing_result = subprocess.run(["python", BRIEFING_SCRIPT], capture_output=True, text=True)
print(briefing_result.stdout)
if briefing_result.stderr:
    print("[WARN] Briefing Errors:\n", briefing_result.stderr)

# Utility functions
def clean_and_tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return [w for w in text.split() if w not in stop_words and len(w) > 3]

def get_top_keywords(texts, n=15):
    words = []
    for text in texts:
        words.extend(clean_and_tokenize(text))
    return Counter(words).most_common(n)

def calculate_nai(left, right):
    if not left or not right:
        return None
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(left + right)
    sim_matrix = cosine_similarity(tfidf[:len(left)], tfidf[len(left):])
    return round(sim_matrix.mean(), 3)

def summarize_side(texts, max_lines=3):
    if not texts:
        return "No coverage."
    text_blob = " ".join(texts)
    tokens = nltk.sent_tokenize(text_blob)
    top_sentences = sorted(tokens, key=lambda s: -len(clean_and_tokenize(s)))[:max_lines]
    return " ".join(wrap(" ".join(top_sentences), width=100))

def deduplicate_summaries(texts, similarity_threshold=0.95):
    unique = []
    for text in texts:
        if not any(SequenceMatcher(None, text, seen).ratio() > similarity_threshold for seen in unique):
            unique.append(text)
    return unique

def render_plot_to_html(fig):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded}"/>'

# Report visuals
def generate_bias_heatmap(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    pivot = df.pivot_table(index='date', columns='topic', values='bias', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(pivot.T.fillna(0), aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([d.strftime('%m-%d') for d in pivot.index], rotation=90)
    plt.colorbar(heatmap)
    plt.title("Bias Heatmap Over Time")
    fig.tight_layout()
    return render_plot_to_html(fig)

def generate_bias_trend_chart(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    trend = df.groupby(['date', 'topic'])['bias'].mean().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for topic in trend.columns:
        ax.plot(trend.index, trend[topic], label=topic)
    plt.title("Bias Trend Over Time")
    plt.legend()
    plt.tight_layout()
    return render_plot_to_html(fig)

def generate_sentiment_trend_chart(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['summary'].fillna('').apply(lambda text: sia.polarity_scores(str(text))['compound'])
    trend = df.groupby(['date', 'topic'])['sentiment'].mean().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for topic in trend.columns:
        ax.plot(trend.index, trend[topic], label=topic)
    plt.title("Sentiment Trend Over Time")
    plt.legend()
    plt.tight_layout()
    return render_plot_to_html(fig)

def generate_bias_sentiment_overlay(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['summary'].fillna('').apply(lambda text: sia.polarity_scores(str(text))['compound'])
    bias_trend = df.groupby(['date', 'topic'])['bias'].mean().unstack().fillna(0)
    sentiment_trend = df.groupby(['date', 'topic'])['sentiment'].mean().unstack().fillna(0)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    for topic in bias_trend.columns:
        ax1.plot(bias_trend.index, bias_trend[topic], label=f"{topic} Bias", linestyle='-', marker='o')
    ax2 = ax1.twinx()
    for topic in sentiment_trend.columns:
        ax2.plot(sentiment_trend.index, sentiment_trend[topic], label=f"{topic} Sentiment", linestyle='--', marker='x', alpha=0.6)
    ax1.set_ylabel("Bias")
    ax2.set_ylabel("Sentiment")
    ax1.set_title("Bias vs. Sentiment Overlay")
    ax1.axhline(0, linestyle='--', color='black', linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    return render_plot_to_html(fig)

def generate_alerts(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['summary'].fillna('').apply(lambda text: sia.polarity_scores(str(text))['compound'])
    recent = df[df['date'] == df['date'].max()]
    alerts = []
    for topic in recent['topic'].unique():
        topic_df = recent[recent['topic'] == topic]
        if topic_df.empty: continue
        bias_avg = topic_df['bias'].mean()
        sent_avg = topic_df['sentiment'].mean()
        if abs(bias_avg) < 1 and abs(sent_avg) > 0.5:
            alerts.append(f"⚠️ Alert: Neutral bias but high emotional sentiment for topic '{topic}'")
    return "<br>".join(alerts) or "✅ No major divergences detected."

def generate_source_breakdown(df):
    source_bias = df.groupby(['source'])['bias'].mean().sort_values()
    ax = source_bias.plot(kind='barh', figsize=(10, 5), title="Average Bias by Source")
    plt.xlabel("Average Bias")
    plt.axvline(0, linestyle='--', color='black')
    plt.tight_layout()
    fig = ax.get_figure()
    return render_plot_to_html(fig)

# Main report generation
def generate_intelligence_report():
    if not os.path.exists(HISTORY_LOG):
        print("❌ briefing_history.csv not found.")
        return

    df = pd.read_csv(HISTORY_LOG)
    df = df.drop_duplicates()  # Remove exact duplicates

    html = f"""
    <html><head><title>Intelligence Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #222; }}
        h2 {{ margin-top: 30px; color: #444; }}
        .section {{ background: #f8f8f8; padding: 15px; margin-bottom: 30px; border-left: 4px solid #aaa; }}
        .keyword-block, .summary-block {{ font-family: monospace; white-space: pre-wrap; }}
    </style></head><body>
    <h1>🧠 Daily Intelligence Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
    <h2>📊 Topic Bias Heatmap</h2>
    {generate_bias_heatmap(df)}
    <hr>
    <h2>📈 Bias Trend by Topic Over Time</h2>
    {generate_bias_trend_chart(df)}
    <hr>
    <h2>💬 Sentiment Trend by Topic Over Time</h2>
    {generate_sentiment_trend_chart(df)}
    <hr>
    <h2>🔀 Bias vs. Sentiment Overlay</h2>
    {generate_bias_sentiment_overlay(df)}
    <hr>
    <h2>🚨 Narrative Divergence Alerts</h2>
    {generate_alerts(df)}
    <hr>
    <h2>📡 Source Bias Breakdown</h2>
    {generate_source_breakdown(df)}
    <hr>
    """

    for topic in sorted(df["topic"].dropna().unique()):
        topic_df = df[df["topic"] == topic]
        left = deduplicate_summaries(topic_df[topic_df["bias"] <= -1]["summary"].tolist())
        right = deduplicate_summaries(topic_df[topic_df["bias"] >= 1]["summary"].tolist())
        if not left or not right:
            continue

        nai = calculate_nai(left, right)
        nai_status = "✅ Strong Alignment" if nai > 0.7 else "🔶 Partial" if nai > 0.3 else "⚠️ Divergence"
        left_kw = get_top_keywords(left)
        right_kw = get_top_keywords(right)
        shared = set(w for w, _ in left_kw) & set(w for w, _ in right_kw)
        unique_left = set(w for w, _ in left_kw) - shared
        unique_right = set(w for w, _ in right_kw) - shared

        html += f"""
        <div class='section'>
        <h2>🧭 Topic: {topic.title()}</h2>
        <p><strong>NAI Score:</strong> {nai} – {nai_status}</p>
        <div class='summary-block'>
        <strong>↙️ Left Summary:</strong><br>{summarize_side(left)}<br><br>
        <strong>↗️ Right Summary:</strong><br>{summarize_side(right)}
        </div><br>
        <div class='keyword-block'>
🔹 Top Left Keywords:
{', '.join([f"{w} ({c})" for w, c in left_kw])}
🔸 Top Right Keywords:
{', '.join([f"{w} ({c})" for w, c in right_kw])}
⚖️ Shared Keywords:
{', '.join(shared) or 'None'}
🔻 Unique to Left:
{', '.join(unique_left) or 'None'}
🔺 Unique to Right:
{', '.join(unique_right) or 'None'}
        </div>
        </div>
        """

    html += "</body></html>"
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Intelligence Report saved to: {OUTPUT_HTML}")

# Execute
generate_intelligence_report()
