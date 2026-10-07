"""HTTP API (FastAPI).

Run:  uvicorn ragqa.api:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ingest import load_corpus
from .llm import make_llm
from .rag import RAGPipeline
from .store import VectorStore

INDEX_DIR = Path(os.getenv("RAGQA_INDEX_DIR", "index"))
LLM_BACKEND = os.getenv("RAGQA_LLM", "extractive")

app = FastAPI(title="ragqa", version="0.1.0", description="Retrieval-augmented question answering over your documents")
_pipeline: RAGPipeline | None = None


class IngestRequest(BaseModel):
    paths: List[str] = Field(..., examples=[["data/sample"]])
    embedder: str = Field("tfidf", pattern="^(tfidf|sbert)$")
    chunk_size: int = Field(180, ge=20, le=2000)
    overlap: int = Field(40, ge=0, le=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(4, ge=1, le=20)


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if not (INDEX_DIR / "chunks.json").exists():
            raise HTTPException(status_code=409, detail="No index yet - call POST /ingest first")
        _pipeline = RAGPipeline(VectorStore.load(INDEX_DIR), make_llm(LLM_BACKEND))
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": (INDEX_DIR / "chunks.json").exists(), "llm": LLM_BACKEND}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _pipeline
    if req.overlap >= req.chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")
    chunks = load_corpus([Path(p) for p in req.paths], req.chunk_size, req.overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No supported documents (.txt, .md, .pdf) found")
    store = VectorStore.build(chunks, req.embedder)
    store.save(INDEX_DIR)
    _pipeline = RAGPipeline(store, make_llm(LLM_BACKEND))
    return {"chunks": len(chunks), "documents": len({c.source for c in chunks}), "embedder": req.embedder}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    pipeline = _get_pipeline()
    pipeline.k = req.k
    return pipeline.ask(req.question).to_dict()
