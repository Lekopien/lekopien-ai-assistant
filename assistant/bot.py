"""The assistant: classify intent, retrieve the best answer, remember the chat."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm import ask_ollama
from .naive_bayes import NaiveBayes
from .tfidf import TfidfIndex

DEFAULT_KB = Path(__file__).resolve().parent.parent / "site" / "knowledge.json"
FALLBACK = ("I'm not sure about that yet. Could you rephrase, or leave your email "
            "and the team will get back to you?")


@dataclass
class Reply:
    text: str
    intent: str
    confidence: float
    source: str  # "retrieval", "llm" or "fallback"


@dataclass
class Assistant:
    kb_path: Path = DEFAULT_KB
    min_score: float = 0.18
    use_llm: bool = False
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        kb = json.loads(Path(self.kb_path).read_text())
        self.entries = kb["entries"]
        self.index = TfidfIndex([" ".join(e["questions"]) for e in self.entries])
        self.classifier = NaiveBayes().fit(
            [(q, e["intent"]) for e in self.entries for q in e["questions"]]
        )

    def ask(self, message: str) -> Reply:
        intent, confidence = self.classifier.predict(message)
        (best, score), *_ = self.index.search(message, k=1)
        entry = self.entries[best]
        if self.use_llm:
            answer = ask_ollama(message, entry["answer"])
            if answer:
                return self._log(message, Reply(answer, intent, confidence, "llm"))
        if score >= self.min_score:
            reply = Reply(entry["answer"], entry["intent"], confidence, "retrieval")
        else:
            reply = Reply(FALLBACK, intent, confidence, "fallback")
        return self._log(message, reply)

    def _log(self, message: str, reply: Reply) -> Reply:
        self.history.append({"user": message, "bot": reply.text, "intent": reply.intent})
        return reply

    def save_history(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.history, indent=2))
