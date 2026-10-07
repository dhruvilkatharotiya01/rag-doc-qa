"""Retrieval-augmented question answering pipeline."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .llm import LLM, NO_ANSWER, ExtractiveLLM
from .store import VectorStore
from .text import content_tokens

__all__ = ["NO_ANSWER", "Answer", "RAGPipeline", "Source", "cited_indices"]


@dataclass
class Source:
    citation: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "latency_ms": round(self.latency_ms, 1),
            "sources": [asdict(s) for s in self.sources],
        }


class RAGPipeline:
    """Retrieve top-k chunks, then let the LLM answer with citations.

    Two guardrails decide whether to abstain instead of letting the model guess:

    * ``min_score``    - the best retrieved chunk must reach a minimum cosine similarity;
    * ``min_coverage`` - a minimum share of the question's content words must appear
      in the retrieved text. Similarity alone was not enough: character n-grams gave
      an off-topic question a score close to real questions, while term coverage
      separated them cleanly on the evaluation set.
    """

    def __init__(
        self,
        store: VectorStore,
        llm: LLM | None = None,
        k: int = 4,
        retrieval: str = "hybrid",
        min_score: float = 0.08,
        min_coverage: float = 0.4,
    ) -> None:
        self.store = store
        self.llm = llm or ExtractiveLLM()
        self.k = k
        self.retrieval = retrieval
        self.min_score = min_score
        self.min_coverage = min_coverage

    @staticmethod
    def term_coverage(question: str, texts: list[str]) -> float:
        q = content_tokens(question)
        if not q:
            return 0.0
        return len(q & content_tokens(" ".join(texts))) / len(q)

    def should_abstain(self, question: str, hits: list) -> bool:
        if not hits:
            return True
        best = max(score for _, score in hits)
        coverage = self.term_coverage(question, [c.text for c, _ in hits])
        return best < self.min_score or coverage < self.min_coverage

    def ask(self, question: str, k: int | None = None) -> Answer:
        start = time.perf_counter()
        hits = self.store.search(question, k=k or self.k, mode=self.retrieval)
        sources = [Source(c.citation(), round(s, 4), c.text) for c, s in hits]

        if self.should_abstain(question, hits):
            text, abstained = NO_ANSWER, True
        else:
            text = self.llm.generate(question, [c.text for c, _ in hits]).strip()
            abstained = text.startswith("I don't know")
        return Answer(question, text, sources, abstained, (time.perf_counter() - start) * 1000)


def cited_indices(answer_text: str) -> list[int]:
    """Return the 1-based passage numbers cited in an answer, e.g. [1][3] -> [1, 3]."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})
