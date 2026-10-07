"""Tests for chunking, retrieval, guardrails, answering, evaluation and the CLI.

Written as unittest.TestCase classes so they run with both
`python -m pytest` and `python -m unittest discover -s tests`.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ragqa.bm25 import BM25
from ragqa.cli import main as cli_main
from ragqa.evaluate import evaluate, load_eval_set
from ragqa.ingest import Chunk, chunk_text, load_corpus, split_sentences
from ragqa.llm import ExtractiveLLM
from ragqa.rag import NO_ANSWER, RAGPipeline, cited_indices
from ragqa.store import RETRIEVAL_MODES, VectorStore
from ragqa.text import content_tokens, tokenize

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
EVAL = ROOT / "eval" / "questions.jsonl"


class TestText(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("What do the planners maximise?"), ["planner", "maximis"])

    def test_content_tokens_is_a_set(self):
        self.assertEqual(content_tokens("robots robots robot"), {"robot"})


class TestChunking(unittest.TestCase):
    def test_overlap_and_coverage_without_punctuation(self):
        words = [f"w{i}" for i in range(100)]
        chunks = chunk_text(" ".join(words), chunk_size=30, overlap=10)
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])  # overlap kept
        self.assertEqual(set(" ".join(chunks).split()), set(words))  # nothing lost

    def test_chunks_end_on_sentence_boundaries(self):
        text = " ".join(f"Sentence number {i} has exactly six words." for i in range(1, 30))
        for chunk in chunk_text(text, chunk_size=40, overlap=10):
            self.assertTrue(chunk.startswith("Sentence"), chunk)
            self.assertTrue(chunk.endswith("."), chunk)
            self.assertLessEqual(len(chunk.split()), 40)

    def test_sentence_overlap(self):
        text = " ".join(f"Sentence number {i} has exactly six words." for i in range(1, 30))
        first, second = chunk_text(text, chunk_size=40, overlap=10)[:2]
        self.assertEqual(split_sentences(first)[-1], split_sentences(second)[0])

    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("one two three", chunk_size=50, overlap=5), ["one two three"])

    def test_empty_text(self):
        self.assertEqual(chunk_text("   \n  "), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=10, overlap=10)

    def test_markdown_heading_does_not_merge_with_paragraph(self):
        chunk = chunk_text("# Title\nBody text here.")[0]
        self.assertEqual(split_sentences(chunk), ["Title.", "Body text here."])

    def test_loads_sample_corpus(self):
        chunks = load_corpus([SAMPLE])
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len({c.source for c in chunks}), 8)


class TestBM25(unittest.TestCase):
    def test_rare_term_wins(self):
        bm25 = BM25(["the robot moves", "the robot uses GAE for advantages", "the car drives"])
        self.assertEqual(int(bm25.scores("What is GAE?").argmax()), 1)

    def test_unknown_terms_score_zero(self):
        self.assertEqual(float(BM25(["alpha beta", "gamma"]).scores("zeta").sum()), 0.0)


class TestStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore.build(load_corpus([SAMPLE]))

    def test_relevant_document_ranks_first_in_every_mode(self):
        for mode in RETRIEVAL_MODES:
            top_chunk, _ = self.store.search("Which polynomial gives smooth lane changes?", k=1, mode=mode)[0]
            self.assertIn("lane_change.md", top_chunk.source, mode)

    def test_dense_scores_sorted_descending(self):
        scores = [s for _, s in self.store.search("Gaussian process kernel", k=5, mode="dense")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_larger_than_corpus(self):
        self.assertEqual(len(self.store.search("robot", k=10_000)), len(self.store.chunks))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.store.search("robot", mode="magic")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(Path(tmp))
            loaded = VectorStore.load(Path(tmp))
            q = "What is RRT*?"
            self.assertEqual(
                [c.chunk_id for c, _ in self.store.search(q, 3)], [c.chunk_id for c, _ in loaded.search(q, 3)]
            )

    def test_load_missing_index(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            VectorStore.load(Path(tmp))

    def test_build_rejects_empty(self):
        with self.assertRaises(ValueError):
            VectorStore.build([])


class TestExtractiveLLM(unittest.TestCase):
    def test_resolves_leading_reference(self):
        ctx = ["SAC stores transitions in a replay buffer for reuse. This makes it very sample efficient."]
        answer = ExtractiveLLM().generate("Why is it sample efficient?", ctx)
        self.assertIn("replay buffer", answer)

    def test_no_overlap_means_no_answer(self):
        self.assertEqual(
            ExtractiveLLM().generate("Quantum chromodynamics?", ["Robots move arms around."]), NO_ANSWER
        )


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

    def test_k_per_call_does_not_change_default(self):
        self.assertEqual(len(self.pipeline.ask("What is time-to-collision?", k=2).sources), 2)
        self.assertEqual(len(self.pipeline.ask("What is time-to-collision?").sources), self.pipeline.k)

    def test_to_dict_is_json_serialisable(self):
        json.dumps(self.pipeline.ask("What is time-to-collision?").to_dict())

    def test_cited_indices(self):
        self.assertEqual(cited_indices("a [2] b [1] c [2]"), [1, 2])


class TestEvaluation(unittest.TestCase):
    def test_eval_meets_baseline(self):
        pipeline = RAGPipeline(VectorStore.build(load_corpus([SAMPLE])), ExtractiveLLM(), k=4)
        report = evaluate(pipeline, load_eval_set(EVAL))
        self.assertGreaterEqual(report["recall@4"], 95)
        self.assertGreaterEqual(report["keyword_hit"], 90)
        self.assertEqual(report["abstain_acc"], 100)
        self.assertEqual(report["false_abstain"], 0)


class TestChunkCitation(unittest.TestCase):
    def test_citation_format(self):
        self.assertEqual(Chunk("t", "docs/a.pdf", 3, page=2).citation(), "a.pdf, p.2#3")


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_ingest_ask_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = self.run_cli("--index", tmp, "ingest", str(SAMPLE))
            self.assertEqual(code, 0)
            self.assertIn("8 documents", out)

            code, out, _ = self.run_cli("--index", tmp, "ask", "What is time-to-collision?", "--json")
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(out)["abstained"])

            code, out, _ = self.run_cli("--index", tmp, "eval", str(EVAL), "--retrieval", "bm25")
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["retrieval"], "bm25")

    def test_missing_index_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self.run_cli("--index", tmp, "ask", "anything?")
            self.assertEqual(code, 1)
            self.assertIn("ingest", err)


if __name__ == "__main__":
    unittest.main()
