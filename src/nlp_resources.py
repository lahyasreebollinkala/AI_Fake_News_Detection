"""NLTK corpus bootstrap.

Kept in its own module so both the training scripts and the Streamlit app can
import :mod:`src.utils` without duplicating download logic. Resources are only
fetched when they are genuinely missing, so a machine that is offline after the
first run keeps working.

If a download fails outright -- no network on a fresh machine -- the module
degrades instead of raising: a built-in stopword list takes over and the
lemmatiser becomes a pass-through. Training and inference both import this
module, so the two stay consistent even in that degraded mode; only accuracy
changes.
"""

from __future__ import annotations

import nltk

REQUIRED_RESOURCES = (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)

#: Smallest useful English stopword set, used only when NLTK is unavailable.
FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does
    doesn doing don down during each few for from further had hadn has hasn have haven
    having he her here hers herself him himself his how i if in into is isn it its
    itself just me more most my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan she should shouldn so some such than
    that the their theirs them themselves then there these they this those through to
    too under until up very was wasn we were weren what when where which while who
    whom why with won would wouldn you your yours yourself yourselves
    """.split()
)


def ensure_nltk_resources() -> list[str]:
    """Download any missing NLTK corpora. Returns the resources that were fetched."""
    downloaded: list[str] = []

    for path, resource in REQUIRED_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            if nltk.download(resource, quiet=True):
                downloaded.append(resource)

    return downloaded


ensure_nltk_resources()


def _load_stopwords() -> frozenset[str]:
    """NLTK's English stopword list, or the built-in fallback."""
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


def _load_lemmatizer():
    """A WordNet lemmatiser, or a pass-through when WordNet is unavailable.

    ``WordNetLemmatizer`` defers its corpus load until the first call, so the
    constructor succeeding proves nothing -- probe it with a real token.
    """
    try:
        from nltk.stem import WordNetLemmatizer

        candidate = WordNetLemmatizer()
        candidate.lemmatize("running")
        return candidate
    except LookupError:
        return None


stop_words: frozenset[str] = _load_stopwords()


class _IdentityLemmatizer:
    """Stand-in that returns its input unchanged."""

    def lemmatize(self, word: str, pos: str = "n") -> str:
        return word


lemmatizer = _load_lemmatizer() or _IdentityLemmatizer()


def degraded() -> bool:
    """``True`` when NLTK resources are unavailable and fallbacks are in use."""
    return isinstance(lemmatizer, _IdentityLemmatizer)


__all__ = [
    "ensure_nltk_resources",
    "stop_words",
    "lemmatizer",
    "degraded",
    "FALLBACK_STOPWORDS",
]
