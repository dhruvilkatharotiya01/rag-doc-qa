"""Offline evaluation of retrieval and answer quality.

Each line of the eval file is JSON:
  {"question": "...", "source": "file.md" | null, "keywords": ["..."]}

``source: null`` marks an unanswerable question: the system should abstain.

Metrics
  recall@k        - expected source appears among the top-k retrieved chunks
  mrr             - mean reciprocal rank of the first chunk from the expected source
  keyword_hit     - answer contains at least one expected keyword
  abstain_acc     - unanswerable questions correctly answered with "I don't know"
  false_abstain   - answerable questions where the system wrongly abstained
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List

from .rag import RAGPipeline


def load_eval_set(path: Path) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def evaluate(pipeline: RAGPipeline, rows: List[dict]) -> dict:
    recall, rr, kw, false_abstain, abstain_ok, latencies = [], [], [], [], [], []
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

        names = [Path(s.citation.split(",")[0].split("#")[0]).name for s in ans.sources]
        hit_rank = next((i + 1 for i, n in enumerate(names) if n == expected), None)
        recall.append(hit_rank is not None)
        rr.append(1.0 / hit_rank if hit_rank else 0.0)
        false_abstain.append(ans.abstained)
        answer_lc = ans.answer.lower()
        kw_hit = any(k.lower() in answer_lc for k in row.get("keywords", []))
        kw.append(kw_hit)
        if hit_rank is None or not kw_hit:
            failures.append({"question": row["question"], "expected": expected, "retrieved": names,
                             "answer": ans.answer[:200]})

    def pct(xs: list) -> float | None:
        return round(100 * mean(xs), 1) if xs else None

    return {
        "n_questions": len(rows),
        f"recall@{pipeline.k}": pct(recall),
        "mrr": round(mean(rr), 3) if rr else None,
        "keyword_hit": pct(kw),
        "abstain_acc": pct(abstain_ok),
        "false_abstain": pct(false_abstain),
        "mean_latency_ms": round(mean(latencies), 1) if latencies else None,
        "failures": failures,
    }
