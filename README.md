# Lekopien AI Assistant

[![test](https://github.com/Lekopien/lekopien-ai-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/Lekopien/lekopien-ai-assistant/actions/workflows/test.yml)

**Live demo: https://ai.lekopien.com**

A customer-support chatbot for a web studio, built from first principles with **no API key, no paid service and no dependencies**. It understands what a visitor wants with a Naive Bayes classifier, finds the best answer with TF-IDF vector search, and can optionally hand off to a free local LLM through [Ollama](https://ollama.com).

The same models are implemented twice: in Python (`assistant/`) for the CLI and tests, and in JavaScript (`site/assistant.js`) so the demo runs fully in the browser.

## Features

- **Intent classification**: multinomial Naive Bayes with Laplace smoothing and a softmax confidence score.
- **Semantic-ish retrieval**: TF-IDF with smoothed IDF and cosine similarity over a JSON knowledge base.
- **Honest fallback**: below a similarity threshold the bot says it doesn't know instead of guessing.
- **Chat history**: every turn is recorded and can be saved to JSON (`--save chat.json`).
- **Optional LLM**: `--llm` asks a local Ollama model, grounding it with the retrieved answer. If Ollama isn't running, it falls back to retrieval automatically.

## Quick start

```bash
git clone https://github.com/Lekopien/lekopien-ai-assistant
cd lekopien-ai-assistant
python3 cli.py                 # chat in the terminal
python3 cli.py --save chat.json
python3 -m unittest -v         # run the tests
```

```
you> my site is slow
bot> We troubleshoot broken, slow or hacked sites. Share the URL and what changed recently...
     [fix_site 35% via retrieval]
```

Use it as a library:

```python
from assistant import Assistant

bot = Assistant()
reply = bot.ask("how much does a website cost?")
print(reply.intent, reply.text)
```

### With a local LLM (optional, free)

```bash
ollama pull llama3.2
python3 cli.py --llm
```

## How it works

```
message ─► tokenize ─┬─► Naive Bayes ─► intent + confidence
                     └─► TF-IDF cosine ─► best entry ─► score ≥ 0.18 ? answer : fallback
```

| File | What it does |
| --- | --- |
| `assistant/text.py` | Tokenizer, stop-words and a small suffix stemmer |
| `assistant/tfidf.py` | TF-IDF index and cosine search |
| `assistant/naive_bayes.py` | Naive Bayes classifier |
| `assistant/bot.py` | Combines the models, keeps history |
| `assistant/llm.py` | Optional Ollama client (stdlib `urllib`) |
| `site/knowledge.json` | The knowledge base. Edit it to teach the bot new answers |
| `site/` | Browser demo deployed at ai.lekopien.com |

## Teach it something new

Add an entry to `site/knowledge.json`:

```json
{"intent": "maintenance", "questions": ["do you offer maintenance", "monthly support plan"],
 "answer": "Yes, we offer monthly maintenance with updates, backups and monitoring."}
```

## Deploy (free)

The `site/` folder is served by a Cloudflare Worker with static assets (`wrangler.jsonc`), on the free plan with no card. Connect the repo in **Cloudflare → Workers & Pages → Create → Import a repository** and it deploys to `ai.lekopien.com` on every push.

## License

MIT
