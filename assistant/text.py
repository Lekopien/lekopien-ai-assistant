"""Tokenising and light normalisation shared by every model."""
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "i", "you", "me", "my", "your", "we",
    "to", "of", "and", "or", "in", "on", "for", "it", "do", "does", "can",
    "could", "would", "be", "with", "what", "how", "this", "that", "please",
}

_WORD = re.compile(r"[a-z0-9']+")


def stem(word: str) -> str:
    """A tiny suffix stripper; good enough to match 'sites' with 'site'."""
    for suffix in ("ing", "ies", "ed", "ly", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]
