"""Command line interface.

ragqa ingest data/sample
ragqa ask "What does SAC optimise?"
ragqa eval eval/questions.jsonl
ragqa benchmark data/sample eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import benchmark, evaluate, load_eval_set, to_markdown
from .ingest import load_corpus
from .llm import make_llm
from .rag import RAGPipeline
from .store import RETRIEVAL_MODES, VectorStore

LLMS = ["extractive", "ollama", "openai"]


def _add_query_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", default="extractive", choices=LLMS, help="answer generator")
    p.add_argument("--retrieval", default="hybrid", choices=RETRIEVAL_MODES, help="ranking method")
    p.add_argument("-k", type=int, default=4, help="number of passages to retrieve")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragqa", description="Question answering over your documents.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--index", default="index", help="index directory (default: index)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="build the index from files or folders")
    p_ing.add_argument("paths", nargs="+")
    p_ing.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p_ing.add_argument("--chunk-size", type=int, default=120, help="words per chunk")
    p_ing.add_argument("--overlap", type=int, default=30, help="words shared by neighbouring chunks")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="print the full answer as JSON")
    _add_query_args(p_ask)

    p_eval = sub.add_parser("eval", help="run the evaluation set against the index")
    p_eval.add_argument("eval_file")
    _add_query_args(p_eval)

    p_bench = sub.add_parser("benchmark", help="compare retrieval modes and chunk sizes")
    p_bench.add_argument("docs")
    p_bench.add_argument("eval_file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = Path(args.index)

    if args.cmd == "ingest":
        try:
            chunks = load_corpus([Path(p) for p in args.paths], args.chunk_size, args.overlap)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not chunks:
            print("error: no supported documents found (.txt, .md, .pdf)", file=sys.stderr)
            return 1
        VectorStore.build(chunks, args.embedder).save(index)
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} documents -> {index}/")
        return 0

    if args.cmd == "benchmark":
        print(to_markdown(benchmark([Path(args.docs)], load_eval_set(Path(args.eval_file)))))
        return 0

    try:
        store = VectorStore.load(index)
        llm = make_llm(args.llm)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pipeline = RAGPipeline(store, llm, k=args.k, retrieval=args.retrieval)

    if args.cmd == "ask":
        ans = pipeline.ask(args.question)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
            return 0
        print(ans.answer)
        print("\nClosest passages (below the confidence threshold):" if ans.abstained else "\nSources:")
        for i, s in enumerate(ans.sources, 1):
            print(f"  [{i}] {s.citation}  (similarity {s.score:.3f})")
        return 0

    report = evaluate(pipeline, load_eval_set(Path(args.eval_file)))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
