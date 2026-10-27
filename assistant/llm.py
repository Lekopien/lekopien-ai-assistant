"""Optional local LLM via Ollama (free, runs on your machine, no API key)."""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ollama(prompt: str, context: str, model: str = "llama3.2", timeout: float = 30) -> str | None:
    body = json.dumps({
        "model": model,
        "stream": False,
        "prompt": f"Answer using this context when relevant.\nContext: {context}\n\nQuestion: {prompt}",
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    except (OSError, ValueError, KeyError):
        return None  # Ollama not running: caller falls back to retrieval.
