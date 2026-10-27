// Browser port of the Python assistant: same tokenizer, TF-IDF and Naive Bayes.
const STOPWORDS = new Set("a an the is are am i you me my your we to of and or in on for it do does can could would be with what how this that please".split(" "));

export function stem(w) {
  for (const s of ["ing", "ies", "ed", "ly", "s"]) {
    if (w.endsWith(s) && !w.endsWith("ss") && w.length - s.length >= 3) {
      return w.slice(0, -s.length) + (s === "ies" ? "y" : "");
    }
  }
  return w;
}

export const tokenize = (text) =>
  (text.toLowerCase().match(/[a-z0-9']+/g) || []).filter((w) => !STOPWORDS.has(w)).map(stem);

const count = (xs) => xs.reduce((m, x) => m.set(x, (m.get(x) || 0) + 1), new Map());

class TfidfIndex {
  constructor(docs) {
    const toks = docs.map(tokenize);
    const df = count(toks.flatMap((t) => [...new Set(t)]));
    this.idf = new Map([...df].map(([t, c]) => [t, Math.log((1 + docs.length) / (1 + c)) + 1]));
    this.vectors = toks.map((t) => this.vectorize(t));
  }
  vectorize(tokens) {
    const vec = new Map();
    for (const [t, c] of count(tokens)) vec.set(t, (c / tokens.length) * (this.idf.get(t) || 0));
    const norm = Math.hypot(...vec.values()) || 1;
    for (const [t, v] of vec) vec.set(t, v / norm);
    return vec;
  }
  best(query) {
    const q = this.vectorize(tokenize(query));
    let best = 0, score = -1;
    this.vectors.forEach((doc, i) => {
      let s = 0;
      for (const [t, w] of q) s += w * (doc.get(t) || 0);
      if (s > score) [best, score] = [i, s];
    });
    return { index: best, score };
  }
}

class NaiveBayes {
  constructor(examples, alpha = 1) {
    this.alpha = alpha;
    this.classes = new Map();
    this.words = new Map();
    this.vocab = new Set();
    for (const [text, label] of examples) {
      this.classes.set(label, (this.classes.get(label) || 0) + 1);
      const wc = this.words.get(label) || new Map();
      for (const t of tokenize(text)) { wc.set(t, (wc.get(t) || 0) + 1); this.vocab.add(t); }
      this.words.set(label, wc);
    }
  }
  predict(text) {
    const tokens = tokenize(text);
    const total = [...this.classes.values()].reduce((a, b) => a + b, 0);
    const logs = [...this.classes].map(([label, n]) => {
      const wc = this.words.get(label);
      const denom = [...wc.values()].reduce((a, b) => a + b, 0) + this.alpha * this.vocab.size;
      return [label, tokens.reduce((lp, t) => lp + Math.log(((wc.get(t) || 0) + this.alpha) / denom), Math.log(n / total))];
    });
    const top = Math.max(...logs.map(([, l]) => l));
    const exp = logs.map(([k, l]) => [k, Math.exp(l - top)]);
    const z = exp.reduce((a, [, e]) => a + e, 0);
    const [label, e] = exp.reduce((a, b) => (b[1] > a[1] ? b : a));
    return { intent: label, confidence: e / z };
  }
}

export const FALLBACK = "I'm not sure about that yet. Could you rephrase, or leave your email and the team will get back to you?";

export class Assistant {
  constructor(kb, minScore = 0.18) {
    this.entries = kb.entries;
    this.minScore = minScore;
    this.index = new TfidfIndex(this.entries.map((e) => e.questions.join(" ")));
    this.nb = new NaiveBayes(this.entries.flatMap((e) => e.questions.map((q) => [q, e.intent])));
  }
  ask(message) {
    const { intent, confidence } = this.nb.predict(message);
    const { index, score } = this.index.best(message);
    if (score >= this.minScore) {
      const e = this.entries[index];
      return { text: e.answer, intent: e.intent, confidence, score, source: "retrieval" };
    }
    return { text: FALLBACK, intent, confidence, score, source: "fallback" };
  }
}
