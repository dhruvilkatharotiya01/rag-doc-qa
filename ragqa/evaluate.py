"""Offline evaluation of retrieval and answer quality.

Each line of the eval file is JSON:
  {"question": "...", "source": "file.md" | null, "keywords": ["..."]}

``source: null`` marks an unanswerable question: the system should abstain.

Metrics
  recall@k        - expected source appears among the top-k retrieved chunks
  top1            - the first retrieved chunk comes from the expected source
  evidence@k      - an expected keyword appears in the retrieved passages (passage-level recall)
  mrr             - mean reciprocal rank of the first chunk from the expected source
  keyword_hit     - answer contains at least one expected keyword
  abstain_acc     - unanswerable questions correctly answered with "I don't know"
  false_abstain   - answerable questions where the system wrongly abstained
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from .ingest import load_corpus
from .llm import ExtractiveLLM
from .rag import RAGPipeline
from .store import RETRIEVAL_MODES, VectorStore


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _source_name(citation: str) -> str:
    return citation.split(",")[0].split("#")[0]


def evaluate(pipeline: RAGPipeline, rows: list[dict]) -> dict:
    recall, top1, evidence, rr, kw = [], [], [], [], []
    false_abstain, abstain_ok, latencies = [], [], []
    failures = []
    for row in rows:
        ans = pipeline.ask(row["question"])
        latencies.append(ans.latency_ms)
        expected = row.get("source")
        if expected is None:
            abstain_ok.append(ans.abstained)
            if not ans.abstained:
                failures.append({"question": row["question"], "issue": "should have abstained"})
            continue

        names = [_source_name(s.citation) for s in ans.sources]
        hit_rank = next((i + 1 for i, n in enumerate(names) if n == expected), None)
        recall.append(hit_rank is not None)
        top1.append(hit_rank == 1)
        context_lc = " ".join(s.text for s in ans.sources).lower()
        evidence.append(any(k.lower() in context_lc for k in row.get("keywords", [])))
        rr.append(1.0 / hit_rank if hit_rank else 0.0)
        false_abstain.append(ans.abstained)
        answer_lc = ans.answer.lower()
        kw_hit = any(k.lower() in answer_lc for k in row.get("keywords", []))
        kw.append(kw_hit)
        if hit_rank is None or not kw_hit:
            failures.append(
                {
                    "question": row["question"],
                    "expected": expected,
                    "retrieved": names,
                    "answer": ans.answer[:200],
                }
            )

    def pct(xs: list) -> float | None:
        return round(100 * mean(xs), 1) if xs else None

    return {
        "n_questions": len(rows),
        "retrieval": pipeline.retrieval,
        f"recall@{pipeline.k}": pct(recall),
        "top1": pct(top1),
        f"evidence@{pipeline.k}": pct(evidence),
        "mrr": round(mean(rr), 3) if rr else None,
        "keyword_hit": pct(kw),
        "abstain_acc": pct(abstain_ok),
        "false_abstain": pct(false_abstain),
        "mean_latency_ms": round(mean(latencies), 1) if latencies else None,
        "failures": failures,
    }


def benchmark(
    doc_paths: list[Path], rows: list[dict], chunk_sizes: tuple[int, ...] = (60, 120, 180), k: int = 4
) -> list[dict]:
    """Evaluate every retrieval mode at several chunk sizes (extractive answers)."""
    results = []
    for size in chunk_sizes:
        store = VectorStore.build(load_corpus(doc_paths, chunk_size=size, overlap=size // 4))
        for mode in RETRIEVAL_MODES:
            report = evaluate(RAGPipeline(store, ExtractiveLLM(), k=k, retrieval=mode), rows)
            report.pop("failures")
            results.append({"chunk_size": size, "chunks": len(store.chunks), **report})
    return results


def to_markdown(results: list[dict]) -> str:
    cols = [c for c in results[0] if c not in ("n_questions",)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in results]
    return "\n".join(lines)
