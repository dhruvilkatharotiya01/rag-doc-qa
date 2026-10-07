"""HTTP API (FastAPI).

Run:  uvicorn ragqa.api:app --reload
Docs: http://localhost:8000/docs

Configuration (environment variables):
  RAGQA_INDEX_DIR   where the index is stored            (default: index)
  RAGQA_DATA_DIR    the only folder /ingest may read from (default: data)
  RAGQA_LLM         extractive | ollama | openai          (default: extractive)
  RAGQA_RETRIEVAL   dense | bm25 | hybrid                 (default: hybrid)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .ingest import load_corpus
from .llm import make_llm
from .rag import RAGPipeline
from .store import VectorStore

INDEX_DIR = Path(os.getenv("RAGQA_INDEX_DIR", "index"))
DATA_DIR = Path(os.getenv("RAGQA_DATA_DIR", "data")).resolve()
LLM_BACKEND = os.getenv("RAGQA_LLM", "extractive")
RETRIEVAL = os.getenv("RAGQA_RETRIEVAL", "hybrid")

app = FastAPI(
    title="ragqa",
    version=__version__,
    description="Retrieval-augmented question answering over your documents, with citations.",
)
_pipeline: RAGPipeline | None = None


class IngestRequest(BaseModel):
    paths: list[str] = Field(
        ..., min_length=1, examples=[["sample"]], description="Files or folders, relative to RAGQA_DATA_DIR"
    )
    embedder: str = Field("tfidf", pattern="^(tfidf|sbert)$")
    chunk_size: int = Field(120, ge=20, le=2000)
    overlap: int = Field(30, ge=0, le=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(4, ge=1, le=20)


def _resolve_inside_data_dir(path: str) -> Path:
    """Only allow paths inside DATA_DIR, so API clients cannot read arbitrary server files."""
    candidate = (DATA_DIR / path).resolve()
    if not candidate.is_relative_to(DATA_DIR):
        raise HTTPException(status_code=400, detail=f"Path must be inside the data directory: {path}")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    return candidate


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if not (INDEX_DIR / "chunks.json").exists():
            raise HTTPException(status_code=409, detail="No index yet - call POST /ingest first")
        _pipeline = RAGPipeline(VectorStore.load(INDEX_DIR), make_llm(LLM_BACKEND), retrieval=RETRIEVAL)
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "index_ready": (INDEX_DIR / "chunks.json").exists(),
        "llm": LLM_BACKEND,
        "retrieval": RETRIEVAL,
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _pipeline
    if req.overlap >= req.chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")
    paths = [_resolve_inside_data_dir(p) for p in req.paths]
    chunks = load_corpus(paths, req.chunk_size, req.overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No supported documents (.txt, .md, .pdf) found")
    store = VectorStore.build(chunks, req.embedder)
    store.save(INDEX_DIR)
    _pipeline = RAGPipeline(store, make_llm(LLM_BACKEND), retrieval=RETRIEVAL)
    return {"chunks": len(chunks), "documents": len({c.source for c in chunks}), "embedder": req.embedder}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return _get_pipeline().ask(req.question, k=req.k).to_dict()
