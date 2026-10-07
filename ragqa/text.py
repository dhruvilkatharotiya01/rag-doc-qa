"""Shared text utilities: tokenisation, stop words and light stemming.

Used by BM25 retrieval, the term-coverage guardrail and the extractive answerer,
so all three agree on what counts as a "content word".
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[a-zA-Z0-9äöüÄÖÜß]+")

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "to",
        "in",
        "and",
        "or",
        "what",
        "which",
        "how",
        "does",
        "do",
        "for",
        "on",
        "with",
        "by",
        "be",
        "it",
        "this",
        "that",
        "as",
        "at",
        "why",
        "who",
        "i",
        "my",
        "me",
        "you",
        "your",
        "can",
        "should",
        "when",
        "where",
        "did",
        "was",
        "between",
        "difference",
        "about",
        "there",
        "their",
        "its",
        "than",
        "then",
        "from",
        "into",
        "if",
        "so",
        "not",
        "no",
        "but",
        "also",
        "has",
        "have",
        "used",
        "use",
        "many",
    }
)


def stem(word: str) -> str:
    """Very light suffix stripping so 'maximises'/'maximise' or 'planners'/'planner' match."""
    for suffix in ("ing", "es", "ed", "s", "e"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-cased, stemmed content words in order (duplicates kept, for term frequencies)."""
    return [stem(w) for w in (t.lower() for t in WORD_RE.findall(text)) if w not in STOPWORDS]


def content_tokens(text: str) -> set[str]:
    """Set of lower-cased, stemmed content words."""
    return set(tokenize(text))
