import re
import string
import pandas as pd
import numpy as np

# pyrefly: ignore [missing-import]
import nltk
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
# pyrefly: ignore [missing-import]
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# Safe NLTK Resource Initialization
# -------------------------------------------------

import nltk

RESOURCES = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for path, resource in RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

EMOTIONAL_WORDS = {
    "shocking", "unbelievable", "secret", "miracle", "outrageous", "terrifying",
    "scandal", "furious", "disaster", "horrible", "amazing", "incredible",
    "devastating", "conspiracy", "exposed", "panic", "banned", "alert"
}

CLICKBAIT_KEYWORDS = {
    "you won't believe", "shocking", "must see", "top 10", "secret", "what happens next",
    "mind blowing", "revealed", "truth about", "don't want you to know", "viral"
}

POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "positive", "success", "benefit", "growth", "official", "verified"}
NEGATIVE_WORDS = {"bad", "worst", "terrible", "fake", "hoax", "false", "disaster", "crisis", "attack", "scam"}


def clean_text(text):
    """
    Clean and preprocess news article text safely.
    """
    if pd.isna(text) or text is None:
        return ""

    text_str = str(text).strip()
    if not text_str:
        return ""

    text_str = text_str.lower()
    text_str = re.sub(r"http\S+|www\S+", "", text_str)
    text_str = re.sub(r"<.*?>", "", text_str)
    text_str = re.sub(r"\d+", "", text_str)
    text_str = text_str.translate(str.maketrans("", "", string.punctuation))
    text_str = re.sub(r"\s+", " ", text_str).strip()

    words = text_str.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word, "v") for word in words]

    return " ".join(words)


def extract_article_metadata(text, title=""):
    """
    Extract heuristic metadata features from raw article text for real-time inference.
    Handles null/None titles and texts safely.
    Returns dict with keys: sentiment_score, clickbait_score, emotional_word_ratio, readability_score, source_trust_score
    """
    safe_text = str(text).strip() if (pd.notna(text) and text is not None) else ""
    safe_title = title.strip() if (pd.notna(title) and title is not None) else ""

    full_str = f"{safe_title} {safe_text}".strip()
    words = full_str.split()
    word_count = max(len(words), 1)

    # 1. Clickbait Score (0 to 1)
    title_lower = safe_title.lower()
    clickbait_matches = sum(1 for kw in CLICKBAIT_KEYWORDS if kw in title_lower or kw in full_str.lower()[:200])
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
    exclamation_count = full_str.count("!")
    clickbait_score = min(1.0, (clickbait_matches * 0.35) + (caps_ratio * 0.4) + (exclamation_count * 0.1) + 0.1)

    # 2. Emotional Word Ratio
    emo_count = sum(1 for w in words if w.lower() in EMOTIONAL_WORDS)
    emotional_word_ratio = min(1.0, emo_count / word_count + (0.05 if "!" in full_str else 0.02))

    # 3. Sentiment Score (-1 to 1)
    pos_count = sum(1 for w in words if w.lower() in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w.lower() in NEGATIVE_WORDS)
    if pos_count + neg_count > 0:
        sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
    else:
        sentiment_score = 0.05

    # 4. Readability Score (0 to 100 approx Flesch Reading Ease)
    sentences = max(len(re.split(r"[.!?]+", full_str)), 1)
    avg_words_per_sentence = word_count / sentences
    readability_score = max(10.0, min(100.0, 206.835 - (1.015 * avg_words_per_sentence) - 15.0))

    # 5. Source Trust Score (Default heuristic 0.75)
    source_trust_score = 0.75

    return {
        "sentiment_score": round(sentiment_score, 4),
        "clickbait_score": round(clickbait_score, 4),
        "emotional_word_ratio": round(emotional_word_ratio, 4),
        "readability_score": round(readability_score, 2),
        "source_trust_score": round(source_trust_score, 2),
    }

