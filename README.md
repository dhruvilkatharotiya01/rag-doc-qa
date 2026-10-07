# ragqa: document Q&A with citations, abstention and evaluation

[![ci](https://github.com/dhruvilkatharotiya01/rag-doc-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvilkatharotiya01/rag-doc-qa/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, transparent **retrieval-augmented generation (RAG)** system. Point it at
a folder of PDFs, Markdown or text files and ask questions. Every answer cites
the passages it came from. When the documents don't contain the answer, the
system **abstains** instead of guessing.

It is built to be measured: an evaluation set and a benchmark command report
retrieval and answer quality for every configuration, so each design decision
below is backed by numbers.

```console
$ ragqa ask "Why does PPO clip the probability ratio?"
PPO computes the probability ratio between the new and the old policy for every sampled
action and clips this ratio to a narrow interval, typically between 0.8 and 1.2. [1]
Clipping removes the incentive to move the policy far away from the one that collected
the data, which prevents destructive updates without the second-order optimisation used
by TRPO. [1]

Sources:
  [1] ppo.md#0  (similarity 0.207)
  [2] ppo.md#1  (similarity 0.057)
  ...

$ ragqa ask "Who won the 2018 football world cup?"
I don't know based on the provided documents.
```

## Features

- **Hybrid retrieval**: TF-IDF vectors (or sentence-transformers) plus BM25, fused with Reciprocal Rank Fusion
- **Sentence-aligned chunking** with overlap, page-aware citations for PDFs
- **Two abstention guardrails**: minimum similarity and query-term coverage
- **Pluggable answer generator**: offline extractive answerer, local LLM via Ollama, or any OpenAI-compatible API
- **Evaluation and benchmarking**: recall@k, MRR, passage evidence, answer keyword hits, abstention accuracy, latency
- **CLI, FastAPI service and Docker image**, tested in CI on Python 3.10 to 3.12

## Architecture

```
files ─► ingest (sentence-aligned chunks) ─► TF-IDF / SBERT vectors ─┐
                                          └► BM25 term index ────────┤
                                                                     ▼
question ────────────────────────────► hybrid search (RRF), top-k passages
                                                     │
                          guardrails: min similarity + query-term coverage
                                                     │
                     answer generator (extractive | Ollama | OpenAI-compatible)
                                                     │
                                       answer + numbered citations
```

| Module | Responsibility |
|---|---|
| `ragqa/ingest.py` | Load `.pdf/.md/.txt`, normalise text, sentence-aligned chunking with overlap |
| `ragqa/embeddings.py` | `tfidf` (word + character n-grams, offline, default) or `sbert` (sentence-transformers) |
| `ragqa/bm25.py` | Okapi BM25 in NumPy, no extra dependency |
| `ragqa/store.py` | Index: dense, BM25 and hybrid (RRF) search; save/load |
| `ragqa/llm.py` | `extractive` (no model), `ollama` (local LLM), `openai` (any compatible endpoint) |
| `ragqa/rag.py` | Pipeline: retrieve, apply guardrails, generate a cited answer |
| `ragqa/evaluate.py` | Metrics and the retrieval/chunk-size benchmark |
| `ragqa/api.py` | FastAPI service: `POST /ingest`, `POST /ask`, `GET /health` |
| `ragqa/cli.py` | `ragqa ingest / ask / eval / benchmark` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ragqa ingest data/sample                        # build the index
ragqa ask "What does the Kalman gain decide?"
ragqa ask "What is time-to-collision?" --json   # full answer with sources as JSON
ragqa eval eval/questions.jsonl                 # evaluation report
ragqa benchmark data/sample eval/questions.jsonl
pytest -q                                       # tests

uvicorn ragqa.api:app --reload                  # API docs at http://localhost:8000/docs
```

Use your own documents with `ragqa ingest path/to/pdfs`. Options worth knowing:
`--retrieval dense|bm25|hybrid`, `-k`, `--chunk-size`, `--overlap`, `--embedder sbert`.

### Using a real LLM

```bash
# local and free: install Ollama (https://ollama.com), then
ollama pull llama3.2:3b
ragqa ask "Why is SAC more sample-efficient than PPO?" --llm ollama

# or any OpenAI-compatible API
export OPENAI_API_KEY=...            # optional: OPENAI_BASE_URL, OPENAI_MODEL
ragqa ask "..." --llm openai
```

The LLM only sees the retrieved passages and is instructed to cite them and to
say "I don't know" when they are not enough. The guardrails run before the LLM
is called, so off-topic questions never reach it.

### API and Docker

```bash
docker build -t ragqa .
docker run -p 8000:8000 ragqa

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is time-to-collision?", "k": 3}'
```

The image ships with the sample index. `POST /ingest` only reads files inside
`RAGQA_DATA_DIR` (default `data/`), so API clients cannot index arbitrary files
on the server. Other settings: `RAGQA_INDEX_DIR`, `RAGQA_LLM`, `RAGQA_RETRIEVAL`.

## Evaluation

`eval/questions.jsonl` holds 37 questions over the eight sample documents
(reinforcement learning, Gaussian processes, Kalman filters, motion planning,
inverse kinematics, ROS 2, lane changes): 32 answerable ones, each with the
expected source file and answer keywords, and 5 off-topic ones that must be
refused. Several questions are deliberately paraphrased so they share few words
with the text.

Default configuration (hybrid retrieval, 120-word chunks, k = 4, extractive answers):

| Metric | Value |
|---|---|
| recall@4 (right document retrieved) | 100 % |
| top-1 from the right document | 100 % |
| evidence@4 (answer text inside retrieved passages) | 100 % |
| MRR | 1.00 |
| answer keyword hit | 96.9 % |
| abstention accuracy (off-topic) | 100 % |
| false abstentions | 0 % |
| mean latency | ~2 ms |

Benchmark (`ragqa benchmark data/sample eval/questions.jsonl`):

| chunk size | retrieval | top-1 | evidence@4 | MRR | keyword hit |
|---|---|---|---|---|---|
| 60 | dense | 96.9 | 96.9 | 0.979 | 90.6 |
| 60 | bm25 | 100 | 96.9 | 1.000 | 96.9 |
| 60 | hybrid | 96.9 | 96.9 | 0.984 | 90.6 |
| 120 | dense | 96.9 | 100 | 0.984 | 93.8 |
| 120 | bm25 | 100 | 100 | 1.000 | 96.9 |
| **120** | **hybrid** | **100** | **100** | **1.000** | **96.9** |
| 180 | dense | 96.9 | 100 | 0.984 | 93.8 |
| 180 | bm25 | 100 | 100 | 1.000 | 96.9 |
| 180 | hybrid | 100 | 100 | 1.000 | 93.8 |

### What the numbers showed

- **Similarity alone is not a reliable "I don't know" signal.** The off-topic
  question "Who won the 2018 football world cup?" reaches a similarity of 0.122
  (character n-grams match "2018" and common word pieces), while the weakest
  answerable question scores 0.129. Query-term coverage separates them clearly:
  at most 0.25 for off-topic questions, at least 0.60 for answerable ones. The
  pipeline therefore abstains below 40 % coverage as well as below a similarity floor.
- **Lexical matching helps with exact terms.** Dense TF-IDF ranked a wrong
  passage first for one question; BM25 and the hybrid ranking fixed it. Hybrid
  is the default because it keeps BM25's precision on rare terms (acronyms such
  as GAE, tf2, TTC) and keeps working when a paraphrase shares few exact words.
- **Sentence-aligned chunks improved answers more than retrieval changes did.**
  With fixed word windows, passages began mid-sentence and the extractive
  answerer returned fragments; aligning chunks to sentences raised the keyword
  hit rate from 87.5 % to 90.6 % at the same retrieval quality.
- **"Why" questions need the neighbouring sentence.** The explanation usually
  follows the claim ("... clips this ratio. Clipping removes the incentive ...")
  or is referenced by it ("This makes it far more sample-efficient ..."). Adding
  the following sentence for "why" questions and the preceding one for sentences
  that open with a reference raised the keyword hit rate to 96.9 %.
- **Remaining failure:** "When should a ROS 2 action be used instead of a
  service?" The answering sentence shares only one content word with the
  question. This is the limit of extractive answering; an LLM backend is the
  intended fix.

**Limits of this evaluation.** The corpus is small and the heuristics were
developed against this question set, so these numbers are optimistic. A
held-out question set over larger, real PDFs is the next step (see roadmap).

## Design decisions

- **TF-IDF as the default embedder** keeps the project offline, fast and
  deterministic, so tests and CI need no model downloads. `--embedder sbert`
  switches to dense sentence embeddings without code changes.
- **Guardrails before generation**: abstaining is decided from retrieval
  signals, not left to the LLM, which makes "I don't know" behaviour testable.
- **NumPy instead of a vector database**: exact search is fast enough for tens
  of thousands of chunks and keeps the ranking logic easy to inspect. Only
  `store.py` would change to use FAISS or a vector database.
- **Index files contain a pickled TF-IDF model**: only load indexes you built yourself.

## Project layout

```
ragqa/               package (see table above)
data/sample/         eight short original documents used for tests and evaluation
eval/questions.jsonl evaluation set
tests/               unit, pipeline, CLI and API tests
Dockerfile           API image with the sample index
.github/workflows/   lint, tests on 3.10-3.12, evaluation report, Docker smoke test
```

## Roadmap

- [x] Hybrid retrieval: BM25 + vectors with reciprocal rank fusion
- [x] Sentence-aligned chunking
- [ ] Evaluate the Ollama backend on the same set, including faithfulness (do the cited passages support the answer?)
- [ ] Cross-encoder re-ranking of the top-k passages
- [ ] Held-out evaluation set built from real PDFs

## License

MIT, see [LICENSE](LICENSE).
