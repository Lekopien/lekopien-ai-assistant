"""Multinomial Naive Bayes classifier with Laplace smoothing."""
import math
from collections import Counter, defaultdict

from .text import tokenize


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, examples: list[tuple[str, str]]) -> "NaiveBayes":
        for text, label in examples:
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.word_counts[label].update(tokens)
            self.vocab.update(tokens)
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        total = sum(self.class_counts.values())
        v = len(self.vocab)
        logs = {}
        for label, count in self.class_counts.items():
            words = self.word_counts[label]
            denom = sum(words.values()) + self.alpha * v
            logp = math.log(count / total)
            for t in tokens:
                logp += math.log((words[t] + self.alpha) / denom)
            logs[label] = logp
        # Softmax over log-probabilities, shifted for numerical stability.
        top = max(logs.values())
        exp = {k: math.exp(v - top) for k, v in logs.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.predict_proba(text)
        label = max(probs, key=probs.get)
        return label, probs[label]
