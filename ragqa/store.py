"""A small, dependency-light vector store.

Vectors live in a NumPy matrix and similarity is cosine (dot product of
L2-normalised vectors). This is plenty for tens of thousands of chunks and keeps
the retrieval logic transparent; swapping in FAISS/Chroma later only touches
this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .embeddings import Embedder, load_embedder, make_embedder
from .ingest import Chunk


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks: List[Chunk], embedder_kind: str = "tfidf") -> "VectorStore":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        embedder = make_embedder(embedder_kind)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        store = cls(embedder)
        store.chunks = list(chunks)
        store.matrix = embedder.embed(texts)
        return store

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        k = max(1, min(k, len(self.chunks)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        (directory / "embedder.bin").write_bytes(self.embedder.dumps())
        meta = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        embedder = load_embedder(meta["embedder"], (directory / "embedder.bin").read_bytes())
        store = cls(embedder)
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.matrix = np.load(directory / "vectors.npy")
        return store
