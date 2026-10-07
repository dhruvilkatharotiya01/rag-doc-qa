"""A small, dependency-light index with dense, lexical and hybrid search.

Vectors live in a NumPy matrix and similarity is cosine (dot product of
L2-normalised vectors). A BM25 index over the same chunks is rebuilt on load
(it is cheap). ``hybrid`` search fuses both rankings with Reciprocal Rank
Fusion. This is plenty for tens of thousands of chunks and keeps retrieval
transparent; swapping in FAISS or a vector database later only touches this
module.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder, load_embedder, make_embedder
from .ingest import Chunk

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
RRF_K = 60  # standard constant from Cormack et al. (2009); damps the influence of top ranks


def _rank(scores: np.ndarray) -> np.ndarray:
    """Rank position (0 = best) of every item."""
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    return ranks


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        self.bm25: BM25 | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks: list[Chunk], embedder_kind: str = "tfidf") -> VectorStore:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        embedder = make_embedder(embedder_kind)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        store = cls(embedder)
        store.chunks = list(chunks)
        store.matrix = embedder.embed(texts)
        store.bm25 = BM25(texts)
        return store

    # ----------------------------------------------------------------- search
    def dense_scores(self, query: str) -> np.ndarray:
        assert self.matrix is not None
        return self.matrix @ self.embedder.embed([query])[0]

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        """Return the top-``k`` chunks with their cosine similarity to the query.

        ``mode`` decides the *ranking*; the returned score is always the dense
        cosine similarity, so guardrail thresholds mean the same in every mode.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; use one of {RETRIEVAL_MODES}")
        if self.matrix is None or not self.chunks:
            return []
        k = max(1, min(k, len(self.chunks)))
        dense = self.dense_scores(query)

        if mode == "dense":
            ranking = dense
        else:
            if self.bm25 is None:
                self.bm25 = BM25([c.text for c in self.chunks])
            lexical = self.bm25.scores(query)
            if mode == "bm25":
                ranking = lexical + 1e-6 * dense  # dense only breaks ties (e.g. all-zero BM25)
            else:
                ranking = 1.0 / (RRF_K + _rank(dense)) + 1.0 / (RRF_K + _rank(lexical))

        top = np.argsort(-ranking, kind="stable")[:k]
        return [(self.chunks[i], float(dense[i])) for i in top]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        (directory / "embedder.bin").write_bytes(self.embedder.dumps())
        meta = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        directory = Path(directory)
        meta_path = directory / "chunks.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No index found in {directory}/ - run 'ragqa ingest <docs>' first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedder = load_embedder(meta["embedder"], (directory / "embedder.bin").read_bytes())
        store = cls(embedder)
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.matrix = np.load(directory / "vectors.npy")
        store.bm25 = BM25([c.text for c in store.chunks])
        return store
