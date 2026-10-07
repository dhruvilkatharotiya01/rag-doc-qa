"""Document loading and chunking.

Supported inputs: .txt, .md and .pdf files. Each document is split into
overlapping, word-based chunks so that an answer which spans a chunk boundary
can still be retrieved from at least one chunk.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    """A retrievable piece of a source document."""

    text: str
    source: str
    chunk_index: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        raw = f"{self.source}|{self.page}|{self.chunk_index}|{self.text[:64]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def citation(self) -> str:
        page = f", p.{self.page}" if self.page is not None else ""
        return f"{Path(self.source).name}{page}#{self.chunk_index}"


def normalise_whitespace(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"\1.", text, flags=re.MULTILINE)  # markdown headings -> sentences
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    """Split text into chunks of ``chunk_size`` words with ``overlap`` words shared.

    Word-based chunking keeps the implementation dependency-free and is a good
    baseline; it can later be swapped for a tokenizer-aware splitter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = normalise_whitespace(text).split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader  # imported lazily so .txt/.md work without pypdf

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_document(path: Path, chunk_size: int = 180, overlap: int = 40) -> List[Chunk]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path}")

    chunks: List[Chunk] = []
    if suffix == ".pdf":
        for page_no, page_text in _read_pdf(path):
            for i, piece in enumerate(chunk_text(page_text, chunk_size, overlap)):
                chunks.append(Chunk(piece, str(path), i, page=page_no))
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(piece, str(path), i))
    return chunks


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield child
        elif p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def load_corpus(paths: Iterable[Path], chunk_size: int = 180, overlap: int = 40) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in iter_files(paths):
        chunks.extend(load_document(path, chunk_size, overlap))
    return chunks
