"""Command line interface.

  python -m ragqa.cli ingest data/sample
  python -m ragqa.cli ask "What does SAC optimise?"
  python -m ragqa.cli eval eval/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, load_eval_set
from .ingest import load_corpus
from .llm import make_llm
from .rag import RAGPipeline
from .store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragqa")
    parser.add_argument("--index", default="index", help="index directory (default: index)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="build the index from files or folders")
    p_ing.add_argument("paths", nargs="+")
    p_ing.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p_ing.add_argument("--chunk-size", type=int, default=180)
    p_ing.add_argument("--overlap", type=int, default=40)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--llm", default="extractive", choices=["extractive", "ollama", "openai"])
    p_ask.add_argument("-k", type=int, default=4)

    p_eval = sub.add_parser("eval", help="run the evaluation set")
    p_eval.add_argument("eval_file")
    p_eval.add_argument("--llm", default="extractive", choices=["extractive", "ollama", "openai"])
    p_eval.add_argument("-k", type=int, default=4)

    args = parser.parse_args(argv)
    index = Path(args.index)

    if args.cmd == "ingest":
        chunks = load_corpus([Path(p) for p in args.paths], args.chunk_size, args.overlap)
        if not chunks:
            print("No supported documents found (.txt, .md, .pdf)", file=sys.stderr)
            return 1
        VectorStore.build(chunks, args.embedder).save(index)
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} documents -> {index}/")
        return 0

    pipeline = RAGPipeline(VectorStore.load(index), make_llm(args.llm), k=args.k)
    if args.cmd == "ask":
        ans = pipeline.ask(args.question)
        print(ans.answer)
        print("\nSources:")
        for i, s in enumerate(ans.sources, 1):
            print(f"  [{i}] {s.citation}  (score {s.score})")
        return 0

    report = evaluate(pipeline, load_eval_set(Path(args.eval_file)))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
