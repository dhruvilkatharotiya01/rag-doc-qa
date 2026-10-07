"""Document loading and chunking.

Supported inputs: .txt, .md and .pdf files. Each document is split into
overlapping, sentence-aligned chunks so that an answer which spans a chunk
boundary can still be retrieved from at least one chunk.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

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
    # markdown headings become sentences, so they never merge with the next paragraph
    text = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"\1.", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (rule-based: end punctuation followed by a capital)."""
    parts = []
    for block in re.split(r"\n\s*\n", text):  # paragraphs never share a sentence
        block = " ".join(block.split())
        parts.extend(s for s in _SENTENCE_END.split(block) if s)
    return parts


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of about ``chunk_size`` words that end on sentence boundaries.

    Whole sentences are packed into a chunk until the next one would exceed
    ``chunk_size`` words; the next chunk then starts with the last sentences of
    the previous one (up to ``overlap`` words), so an answer that spans a
    boundary is still retrievable. A single sentence longer than ``chunk_size``
    is split by words. Keeping sentences intact means retrieved passages - and
    extractive answers - never start or end mid-sentence.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    sentences: list[list[str]] = []
    for sent in split_sentences(normalise_whitespace(text)):
        words = sent.split()
        step = chunk_size - overlap
        if len(words) <= chunk_size:
            sentences.append(words)
        else:  # rare: one huge "sentence" (tables, lists without punctuation)
            for start in range(0, len(words), step):
                sentences.append(words[start : start + chunk_size])
                if start + chunk_size >= len(words):
                    break
    if not sentences:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        current: list[list[str]] = []
        n_words = 0
        j = i
        while j < len(sentences) and (not current or n_words + len(sentences[j]) <= chunk_size):
            current.append(sentences[j])
            n_words += len(sentences[j])
            j += 1
        chunks.append(" ".join(" ".join(s) for s in current))
        if j >= len(sentences):
            break
        # step back over trailing sentences that fit in the overlap budget
        back, kept = j, 0
        while back - 1 > i and kept + len(sentences[back - 1]) <= overlap:
            back -= 1
            kept += len(sentences[back])
        i = back
    return chunks


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader  # imported lazily so .txt/.md work without pypdf

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_document(path: Path, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path}")

    chunks: list[Chunk] = []
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


def load_corpus(paths: Iterable[Path], chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_files(paths):
        chunks.extend(load_document(path, chunk_size, overlap))
    return chunks
