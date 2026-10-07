"""Tests for chunking, retrieval, guardrails and evaluation.

Written as unittest.TestCase classes so they run with both
`python -m pytest` and `python -m unittest`.
"""
import json
import tempfile
import unittest
from pathlib import Path

from ragqa.evaluate import evaluate, load_eval_set
from ragqa.ingest import Chunk, chunk_text, load_corpus
from ragqa.llm import ExtractiveLLM
from ragqa.rag import NO_ANSWER, RAGPipeline, cited_indices
from ragqa.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
EVAL = ROOT / "eval" / "questions.jsonl"


class TestChunking(unittest.TestCase):
    def test_overlap_and_coverage(self):
        words = [f"w{i}" for i in range(100)]
        chunks = chunk_text(" ".join(words), chunk_size=30, overlap=10)
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])  # overlap kept
        covered = set(" ".join(chunks).split())
        self.assertEqual(covered, set(words))  # nothing lost

    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("one two three", chunk_size=50, overlap=5), ["one two three"])

    def test_empty_text(self):
        self.assertEqual(chunk_text("   \n  "), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=10, overlap=10)

    def test_loads_sample_corpus(self):
        chunks = load_corpus([SAMPLE])
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len({c.source for c in chunks}), 4)


class TestStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore.build(load_corpus([SAMPLE], chunk_size=120, overlap=30))

    def test_relevant_document_ranks_first(self):
        top_chunk, _ = self.store.search("Which polynomial gives smooth lane changes?", k=1)[0]
        self.assertIn("lane_change.md", top_chunk.source)

    def test_scores_sorted_descending(self):
        scores = [s for _, s in self.store.search("Gaussian process kernel", k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(Path(tmp))
            loaded = VectorStore.load(Path(tmp))
            q = "What is RRT*?"
            self.assertEqual([c.chunk_id for c, _ in self.store.search(q, 3)],
                             [c.chunk_id for c, _ in loaded.search(q, 3)])

    def test_build_rejects_empty(self):
        with self.assertRaises(ValueError):
            VectorStore.build([])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RAGPipeline(VectorStore.build(load_corpus([SAMPLE])), ExtractiveLLM())

    def test_answers_with_citation(self):
        ans = self.pipeline.ask("Besides the expected return, what does SAC maximise?")
        self.assertFalse(ans.abstained)
        self.assertIn("entropy", ans.answer.lower())
        self.assertTrue(cited_indices(ans.answer))

    def test_abstains_on_off_topic_question(self):
        ans = self.pipeline.ask("How do I bake sourdough bread?")
        self.assertTrue(ans.abstained)
        self.assertEqual(ans.answer, NO_ANSWER)

    def test_to_dict_is_json_serialisable(self):
        json.dumps(self.pipeline.ask("What is time-to-collision?").to_dict())

    def test_cited_indices(self):
        self.assertEqual(cited_indices("a [2] b [1] c [2]"), [1, 2])


class TestEvaluation(unittest.TestCase):
    def test_eval_meets_baseline(self):
        pipeline = RAGPipeline(VectorStore.build(load_corpus([SAMPLE])), ExtractiveLLM(), k=4)
        report = evaluate(pipeline, load_eval_set(EVAL))
        self.assertGreaterEqual(report["recall@4"], 90)
        self.assertEqual(report["abstain_acc"], 100)
        self.assertEqual(report["false_abstain"], 0)


class TestChunkCitation(unittest.TestCase):
    def test_citation_format(self):
        self.assertEqual(Chunk("t", "docs/a.pdf", 3, page=2).citation(), "a.pdf, p.2#3")


if __name__ == "__main__":
    unittest.main()
