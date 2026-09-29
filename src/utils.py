"""Shared text-processing and feature-engineering helpers.

Two distinct concerns live here:

* :func:`clean_text` -- the NLP pipeline (lowercasing, URL/HTML/number removal,
  punctuation stripping, stopword removal, lemmatisation) that turns raw article
  copy into TF-IDF-ready tokens.
* :func:`extract_article_metadata` -- transparent, rule-based editorial signals
  (clickbait, emotion, sentiment, readability, source trust) computed directly
  from the raw article. These power the *content signals* panel in the app.
"""

from __future__ import annotations

import re
import string
import unicodedata

import pandas as pd

from src.nlp_resources import lemmatizer, stop_words

# --------------------------------------------------------------------------
# Word lists used by the editorial-signal heuristics
# --------------------------------------------------------------------------

EMOTIONAL_WORDS = {
    "shocking", "unbelievable", "secret", "miracle", "outrageous", "terrifying",
    "scandal", "furious", "disaster", "horrible", "amazing", "incredible",
    "devastating", "conspiracy", "exposed", "panic", "banned", "alert",
    "breaking", "urgent", "warning", "exclusive", "sensational", "outrage",
}

CLICKBAIT_KEYWORDS = {
    "you won't believe", "shocking", "must see", "top 10", "secret",
    "what happens next", "mind blowing", "revealed", "truth about",
    "don't want you to know", "viral", "this is why", "nobody is talking",
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "positive", "success", "benefit",
    "growth", "official", "verified", "agreement", "approved", "recovered",
    "reform", "improve", "launched", "resolved", "increase",
}

NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "fake", "hoax", "false", "disaster", "crisis",
    "attack", "scam", "fraud", "illegal", "reject", "failed", "warning",
    "threat", "collapse", "denied", "shortage", "delay",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_DIGIT_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _to_ascii(text: str) -> str:
    """Fold accents and drop non-Latin characters.

    News copy is full of typographic characters -- curly apostrophes, en and em
    dashes, ellipses, accented loanwords. ``string.punctuation`` only removes the
    32 ASCII marks, so a naive strip leaves ``country's`` and ``countrys`` as
    two different tokens. Decomposing to ASCII first folds every accented letter
    to its base form and removes the typographic marks entirely, which is what
    the vectoriser's ``strip_accents`` would otherwise have to do downstream.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("ascii")



# --------------------------------------------------------------------------
# NLP cleaning
# --------------------------------------------------------------------------

def clean_text(text: object) -> str:
    """Normalise and clean raw article text for vectorisation.

    The function is defensive: ``None``, ``NaN`` and non-string values all
    resolve to an empty string instead of raising, so it can be mapped over
    a dataframe column or called on a single string from the app.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = _to_ascii(cleaned.lower())
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _HTML_RE.sub(" ", cleaned)
    cleaned = _DIGIT_RE.sub(" ", cleaned)
    cleaned = cleaned.translate(_PUNCT_TABLE)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    tokens = [
        lemmatizer.lemmatize(token)
        for token in cleaned.split()
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(tokens)


def clean_series(series: pd.Series) -> pd.Series:
    """Vectorised convenience wrapper around :func:`clean_text`."""
    return series.fillna("").astype(str).map(clean_text)


def build_full_text(title: object, body: object) -> str:
    """Join a headline and body into the single string the model consumes."""
    head = "" if title is None or pd.isna(title) else str(title).strip()
    content = "" if body is None or pd.isna(body) else str(body).strip()
    return f"{head}. {content}".strip() if head else content


_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_SILENT_TAIL_RE = re.compile(r"(?:[^aeiou]e)$")


def count_syllables(word: str) -> int:
    """Approximate English syllable count for a single token.

    A compact heuristic: count vowel groups, drop a silent trailing ``e`` and
    clamp to at least one. Accurate enough for a readability index and far
    cheaper than a pronunciation dictionary.
    """
    token = _VOWEL_GROUP_RE.sub("", word.lower())
    if not token:
        return 0

    groups = len(_VOWEL_GROUP_RE.findall(word.lower()))
    if _SILENT_TAIL_RE.search(word.lower()) and groups > 1:
        groups -= 1

    return max(1, groups)


# --------------------------------------------------------------------------
# Editorial / heuristic signals
# --------------------------------------------------------------------------

def extract_article_metadata(text: object, title: object = "") -> dict:
    """Compute transparent editorial signals for one article.

    These are deliberately simple, auditable heuristics rather than a black
    box, which is why they are surfaced next to the model verdict in the app.

    Returns a dict with the five engineered feature columns.
    """
    safe_text = "" if text is None or pd.isna(text) else str(text).strip()
    safe_title = "" if title is None or pd.isna(title) else str(title).strip()

    full_str = f"{safe_title} {safe_text}".strip()
    if not full_str:
        return {
            "sentiment_score": 0.0,
            "clickbait_score": 0.0,
            "emotional_word_ratio": 0.0,
            "readability_score": 0.0,
            "source_trust_score": 0.0,
        }

    words = full_str.split()
    word_count = max(len(words), 1)
    lowered = full_str.lower()
    title_lower = safe_title.lower()

    # 1. Clickbait score in [0, 1] -- phrase hits, shouting and exclamation.
    phrase_hits = sum(
        1 for phrase in CLICKBAIT_KEYWORDS
        if phrase in title_lower or phrase in lowered[:200]
    )
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
    exclamations = full_str.count("!")
    clickbait_score = min(
        1.0, phrase_hits * 0.35 + caps_ratio * 0.4 + exclamations * 0.1
    )

    # 2. Emotional word ratio in [0, 1].
    emotional_hits = sum(1 for w in words if w.lower() in EMOTIONAL_WORDS)
    emotional_word_ratio = min(
        1.0, emotional_hits / word_count + (0.05 if "!" in full_str else 0.02)
    )

    # 3. Sentiment polarity in [-1, 1].
    positive_hits = sum(1 for w in words if w.lower() in POSITIVE_WORDS)
    negative_hits = sum(1 for w in words if w.lower() in NEGATIVE_WORDS)
    if positive_hits + negative_hits:
        sentiment_score = (positive_hits - negative_hits) / (positive_hits + negative_hits)
    else:
        sentiment_score = 0.0

    # 4. Readability -- Flesch Reading Ease, clamped to [0, 100].
    sentences = max(len(_SENTENCE_SPLIT_RE.split(full_str)), 1)
    words_per_sentence = word_count / sentences
    syllables_per_word = sum(count_syllables(w) for w in words) / word_count
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    readability_score = max(0.0, min(100.0, flesch))

    # 5. Source trust -- neutral prior; the corpus carries no verifiable
    #    publisher identity, so a constant is the honest default.
    source_trust_score = 0.75

    return {
        "sentiment_score": round(sentiment_score, 4),
        "clickbait_score": round(clickbait_score, 4),
        "emotional_word_ratio": round(emotional_word_ratio, 4),
        "readability_score": round(readability_score, 2),
        "source_trust_score": round(source_trust_score, 2),
    }


def text_statistics(text: str) -> dict:
    """Word/character/sentence counts shown alongside a prediction."""
    if not text:
        return {"words": 0, "characters": 0, "sentences": 0, "avg_words_per_sentence": 0.0}

    words = text.split()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    return {
        "words": len(words),
        "characters": len(text),
        "sentences": len(sentences),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1),
    }
