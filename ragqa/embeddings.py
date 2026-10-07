"""Embedding back-ends.

``tfidf``  - scikit-learn TF-IDF (word + character n-grams). Runs fully offline
             and is the default, which keeps tests and CI fast.
``sbert``  - dense sentence-transformers model (e.g. all-MiniLM-L6-v2). Better
             semantic matching; requires ``pip install sentence-transformers``.

Both return L2-normalised vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

import pickle
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def fit(self, texts: List[str]) -> "Embedder": ...

    def embed(self, texts: List[str]) -> np.ndarray: ...

    def dumps(self) -> bytes: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ]
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted before embedding")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalise(matrix)

    def dumps(self) -> bytes:
        return pickle.dumps(self._vectorizer)

    @classmethod
    def loads(cls, blob: bytes) -> "TfidfEmbedder":
        obj = cls.__new__(cls)
        obj._vectorizer = pickle.loads(blob)
        obj._fitted = True
        return obj


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install sentence-transformers to use the 'sbert' embedder") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> "SbertEmbedder":
        return self  # pre-trained, nothing to fit

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalise(vectors.astype(np.float32))

    def dumps(self) -> bytes:
        return self.model_name.encode("utf-8")

    @classmethod
    def loads(cls, blob: bytes) -> "SbertEmbedder":
        return cls(blob.decode("utf-8"))


def make_embedder(kind: str = "tfidf") -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "sbert":
        return SbertEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r} (use 'tfidf' or 'sbert')")


def load_embedder(kind: str, blob: bytes) -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder.loads(blob)
    if kind == "sbert":
        return SbertEmbedder.loads(blob)
    raise ValueError(f"Unknown embedder: {kind!r}")
