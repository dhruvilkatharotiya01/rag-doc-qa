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
from typing import Protocol

from .text import content_tokens

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the context does not contain the answer, reply exactly: "
    f'"{NO_ANSWER}"'
)


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: list[str]) -> str: ...


def build_user_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{numbered}\n\nQuestion: {question}\nAnswer:"


_ANAPHORA = re.compile(r"^(This|These|That|Those|It|Such|Thus|Therefore)\b")


class ExtractiveLLM:
    """Answers with the best-matching sentences from the context. No model needed.

    Sentences are scored by IDF-weighted overlap with the question, with a small
    preference for higher-ranked passages. Two discourse rules keep answers
    readable: a sentence that opens with a reference ("This makes it ...") is
    preceded by the sentence it refers to, and for "why" questions the sentence
    after the best match is added, because explanations usually follow the claim.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_relative_score: float = 0.5) -> None:
        self.max_sentences = max_sentences
        self.min_relative_score = min_relative_score  # extra sentences must score >= this x best

    @staticmethod
    def _sentences(contexts: list[str]) -> list[tuple[int, list[str]]]:
        """Split each passage into sentences, dropping ones already seen in a higher-ranked passage."""
        seen: set[str] = set()
        out = []
        for ci, ctx in enumerate(contexts):
            kept = []
            for sent in re.split(r"(?<=[.!?])\s+", ctx.strip()):
                key = sent.lower()
                if sent and key not in seen:
                    seen.add(key)
                    kept.append(sent)
            out.append((ci, kept))
        return out

    def generate(self, question: str, contexts: list[str]) -> str:
        q = content_tokens(question)
        passages = self._sentences(contexts)
        # (passage index, sentence index, sentence, tokens); titles are too short to answer
        candidates = [
            (ci, si, s, content_tokens(s))
            for ci, sents in passages
            for si, s in enumerate(sents)
            if len(s.split()) >= 5
        ]
        if not candidates:
            return NO_ANSWER

        # IDF over candidate sentences: words that appear everywhere (e.g. the topic name)
        # say little about which sentence actually answers the question.
        n = len(candidates)
        df: dict[str, int] = {}
        for *_, toks in candidates:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        scored = []
        for ci, si, _, toks in candidates:
            score = sum(math.log(1 + n / df[t]) for t in q & toks) / (1 + 0.25 * ci)
            if score > 0:
                scored.append((score, -ci, -si, ci, si))
        if not scored:
            return NO_ANSWER
        scored.sort(reverse=True)

        is_why = bool(re.match(r"\s*why\b", question, re.IGNORECASE))
        picked: list[tuple[int, int]] = []

        def add(ci: int, si: int) -> None:
            if (ci, si) not in picked:
                picked.append((ci, si))

        best = scored[0][0]
        for score, *_, ci, si in scored:
            if len(picked) >= self.max_sentences or score < self.min_relative_score * best:
                break
            sents = passages[ci][1]
            if si > 0 and _ANAPHORA.match(sents[si]):
                add(ci, si - 1)
            add(ci, si)
            if is_why and si + 1 < len(sents):
                add(ci, si + 1)
        # passages in order of their best sentence, sentences in reading order within a passage
        first = {}
        for i, (ci, _) in enumerate(picked):
            first.setdefault(ci, i)
        picked.sort(key=lambda p: (first[p[0]], p[1]))
        return " ".join(f"{passages[ci][1][si]} [{ci + 1}]" for ci, si in picked)


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-configured endpoint
        return json.loads(resp.read().decode("utf-8"))


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, question: str, contexts: list[str]) -> str:
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

    def generate(self, question: str, contexts: list[str]) -> str:
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
