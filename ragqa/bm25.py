"""Okapi BM25 lexical retrieval, implemented with NumPy (no extra dependency).

BM25 complements vector similarity: it rewards exact matches of rare terms
(names, acronyms such as "GAE" or "tf2") that embeddings can blur.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 needs at least one document")
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d)) for d in documents]
        self._len = np.array([sum(tf.values()) for tf in self._tf], dtype=np.float32)
        self._avg_len = float(self._len.mean()) or 1.0
        n = len(documents)
        df = Counter(t for tf in self._tf for t in tf)
        # BM25+ style IDF floor at 0 so very common terms never count negatively
        self.idf = {t: max(0.0, math.log((n - f + 0.5) / (f + 0.5) + 1.0)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (higher is better)."""
        out = np.zeros(len(self._tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self._len / self._avg_len)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if not idf:
                continue
            tf = np.array([d.get(term, 0) for d in self._tf], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
