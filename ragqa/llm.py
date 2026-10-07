"""Answer-generation back-ends.

``extractive`` - no LLM at all: returns the sentences from the retrieved chunks
                 that overlap most with the question. Works offline and is the
                 default so the project runs anywhere.
``ollama``     - local model served by Ollama (http://localhost:11434).
``openai``     - any OpenAI-compatible chat-completions endpoint
                 (OPENAI_API_KEY, optional OPENAI_BASE_URL).

Only the standard library is used for HTTP to keep dependencies small.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from typing import List, Protocol

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\""
)


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: List[str]) -> str: ...


def build_user_prompt(question: str, contexts: List[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{numbered}\n\nQuestion: {question}\nAnswer:"


_WORD = re.compile(r"[a-zA-Z0-9äöüÄÖÜß]+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what", "which", "how",
    "does", "do", "for", "on", "with", "by", "be", "it", "this", "that", "as", "at", "why",
    "who", "i", "my", "me", "you", "your", "can", "should", "when", "where", "did", "was",
    "between", "difference", "about", "there", "their", "its", "than", "then",
}


def _stem(word: str) -> str:
    """Very light suffix stripping so 'maximises'/'maximise' or 'planners'/'planner' match."""
    for suffix in ("ing", "es", "ed", "s", "e"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> set[str]:
    """Lower-cased, lightly stemmed content words (stop words removed)."""
    return {_stem(w.lower()) for w in _WORD.findall(text) if w.lower() not in _STOP}


_tokens = content_tokens


class ExtractiveLLM:
    """Picks the best-matching sentences from the context. No model needed."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: List[str]) -> str:
        q = _tokens(question)
        candidates, seen = [], set()
        for ci, ctx in enumerate(contexts):
            for sent in re.split(r"(?<=[.!?])\s+", ctx):
                sent = sent.strip()
                key = sent.lower()
                # overlapping chunks repeat (possibly truncated) sentences; titles are too short to answer
                if len(sent.split()) < 5 or any(key in s or s in key for s in seen):
                    continue
                seen.add(key)
                candidates.append((ci, sent, _tokens(sent)))
        if not candidates:
            return "I don't know based on the provided documents."

        # IDF over candidate sentences: words that appear everywhere (e.g. the topic name)
        # say little about which sentence actually answers the question.
        n = len(candidates)
        df: dict[str, int] = {}
        for _, _, toks in candidates:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        scored = []
        for ci, sent, toks in candidates:
            score = sum(math.log(1 + n / df[t]) for t in q & toks)
            score /= 1 + 0.25 * ci  # prefer sentences from higher-ranked chunks
            if score > 0:
                scored.append((score, -ci, sent, ci))
        if not scored:
            return "I don't know based on the provided documents."
        scored.sort(reverse=True)
        picked = scored[: self.max_sentences]
        return " ".join(f"{s} [{ci + 1}]" for _, _, s, ci in picked)


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-configured endpoint
        return json.loads(resp.read().decode("utf-8"))


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, question: str, contexts: List[str]) -> str:
        out = _post_json(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts)},
                ],
            },
        )
        return out["message"]["content"].strip()


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use the 'openai' backend")

    def generate(self, question: str, contexts: List[str]) -> str:
        out = _post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts)},
                ],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return out["choices"][0]["message"]["content"].strip()


def make_llm(kind: str = "extractive") -> LLM:
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "ollama":
        return OllamaLLM()
    if kind == "openai":
        return OpenAICompatLLM()
    raise ValueError(f"Unknown LLM backend: {kind!r} (use extractive, ollama or openai)")
