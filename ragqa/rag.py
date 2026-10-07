"""Retrieval-augmented question answering pipeline."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List

from .llm import LLM, ExtractiveLLM, content_tokens
from .store import VectorStore

NO_ANSWER = "I don't know based on the provided documents."


@dataclass
class Source:
    citation: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "latency_ms": round(self.latency_ms, 1),
            "sources": [s.__dict__ for s in self.sources],
        }


class RAGPipeline:
    """Retrieve top-k chunks, then let the LLM answer with citations.

    Two guardrails decide whether to abstain instead of letting the model guess:
    * ``min_score``    - the best chunk must reach a minimum cosine similarity;
    * ``min_coverage`` - a minimum share of the question's content words must
      appear in the retrieved text. Similarity alone was not enough: character
      n-grams gave an off-topic question a score close to real questions, while
      term coverage separated them cleanly on the evaluation set.
    """

    def __init__(self, store: VectorStore, llm: LLM | None = None, k: int = 4,
                 min_score: float = 0.08, min_coverage: float = 0.4) -> None:
        self.store = store
        self.llm = llm or ExtractiveLLM()
        self.k = k
        self.min_score = min_score
        self.min_coverage = min_coverage

    @staticmethod
    def term_coverage(question: str, texts: List[str]) -> float:
        q = content_tokens(question)
        if not q:
            return 0.0
        return len(q & content_tokens(" ".join(texts))) / len(q)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.store.search(question, k=self.k)
        sources = [Source(c.citation(), round(s, 4), c.text) for c, s in hits]

        if (not hits or hits[0][1] < self.min_score
                or self.term_coverage(question, [c.text for c, _ in hits]) < self.min_coverage):
            return Answer(question, NO_ANSWER, sources, abstained=True,
                          latency_ms=(time.perf_counter() - start) * 1000)

        text = self.llm.generate(question, [c.text for c, _ in hits])
        abstained = text.strip().startswith("I don't know")
        return Answer(question, text, sources, abstained=abstained,
                      latency_ms=(time.perf_counter() - start) * 1000)


def cited_indices(answer_text: str) -> List[int]:
    """Return the 1-based passage numbers cited in an answer, e.g. [1][3] -> [1, 3]."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})
