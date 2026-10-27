"""TF-IDF vector space model with cosine similarity, written from scratch."""
import math
from collections import Counter

from .text import tokenize


class TfidfIndex:
    def __init__(self, documents: list[str]):
        self.documents = documents
        tokenized = [tokenize(d) for d in documents]
        df = Counter(term for tokens in tokenized for term in set(tokens))
        n = len(documents)
        # Smoothed IDF so a term found in every document still counts a little.
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vectorize(tokens) for tokens in tokenized]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in counts.items()} if tokens else {}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q = self._vectorize(tokenize(query))
        scores = [
            (i, sum(w * doc.get(t, 0.0) for t, w in q.items()))
            for i, doc in enumerate(self.vectors)
        ]
        scores.sort(key=lambda s: s[1], reverse=True)
        return scores[:k]
