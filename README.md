# ragqa: document Q&A with retrieval, citations and evaluation

A small, transparent **retrieval-augmented generation (RAG)** system. Point it at
a folder of PDFs, Markdown or text files and ask questions. Every answer cites
the passages it came from. When the documents don't contain the answer, the
system **abstains** instead of guessing.

It is built to be measured: an evaluation set and a CLI command report
retrieval recall, MRR, answer keyword hits and abstention accuracy, so every
change can be checked against numbers.

```
python -m ragqa.cli ask "Why does SAC use two Q-functions?"

SAC learns three kinds of networks: a stochastic policy (the actor), and two
Q-functions (the critics). [1] Using two critics and taking the minimum of their
estimates reduces the overestimation bias that is common in Q-learning. [1]

Sources:
  [1] soft_actor_critic.md#0  (score 0.157)
  ...
```

## Architecture

```
files ─► ingest (chunking with overlap) ─► embeddings ─► vector store (NumPy, cosine)
                                                            │
question ───────────────────────────────► retrieve top-k ◄──┘
                                              │
                         guardrails (min similarity, query-term coverage)
                                              │
                       answer generator (extractive | Ollama | OpenAI-compatible)
                                              │
                                answer + numbered citations
```

| Module | Responsibility |
|---|---|
| `ragqa/ingest.py` | Load `.pdf/.md/.txt`, normalise text, word-based chunking with overlap, page-aware citations |
| `ragqa/embeddings.py` | `tfidf` (word + char n-grams, offline, default) or `sbert` (sentence-transformers) |
| `ragqa/store.py` | NumPy vector store, cosine search, save/load |
| `ragqa/llm.py` | `extractive` (no model, offline), `ollama` (local LLM), `openai` (any compatible endpoint) |
| `ragqa/rag.py` | Pipeline: retrieve, apply guardrails, generate a cited answer |
| `ragqa/evaluate.py` | recall@k, MRR, keyword hit rate, abstention accuracy, latency |
| `ragqa/api.py` | FastAPI service: `POST /ingest`, `POST /ask`, `GET /health` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ragqa.cli ingest data/sample            # build the index
python -m ragqa.cli ask "What is time-to-collision?"
python -m ragqa.cli eval eval/questions.jsonl     # evaluation report
python -m pytest -q                               # tests

uvicorn ragqa.api:app --reload                    # API docs at http://localhost:8000/docs
```

Use your own documents: `python -m ragqa.cli ingest path/to/pdfs`.

### Using a real LLM

```bash
# local, free: install Ollama (https://ollama.com), then
ollama pull llama3.2:3b
python -m ragqa.cli ask "Why is SAC sample-efficient?" --llm ollama

# or any OpenAI-compatible API
export OPENAI_API_KEY=...            # optional: OPENAI_BASE_URL, OPENAI_MODEL
python -m ragqa.cli ask "..." --llm openai
```

The LLM backends only see the retrieved passages and are prompted to cite them
and to say "I don't know" when the context is insufficient.

### Docker

```bash
docker build -t ragqa .
docker run -p 8000:8000 ragqa
```

## Evaluation

`eval/questions.jsonl` holds 18 questions over the sample documents: 15
answerable ones (each with the expected source file and answer keywords) and 3
off-topic ones that must be refused.

Baseline (TF-IDF retrieval, extractive answers, k = 4):

| Metric | Value |
|---|---|
| recall@4 | 100 % |
| MRR | 1.00 |
| answer keyword hit | 86.7 % |
| abstention accuracy (off-topic) | 100 % |
| false abstentions | 0 % |
| mean latency | ~2 ms |

What the numbers showed while building it:

* **Similarity alone was not a reliable "I don't know" signal.** With
  character n-grams, an off-topic question ("Who won the 2018 World Cup?")
  scored 0.119, almost the same as a real question (0.122). The share of
  question terms found in the retrieved text separated them cleanly (≥ 0.67 vs
  ≤ 0.17), so the pipeline uses term coverage as a second guardrail.
* **Overlapping chunks produced duplicated answer sentences.** Fixed by
  de-duplicating (including truncated) sentences.
* **Topic words dominated sentence selection.** Weighting overlap by IDF across
  candidate sentences, plus a small preference for higher-ranked chunks, fixed
  answers that just repeated the topic definition.
* **Remaining failures are "why" questions** whose answer sits in a causal
  sentence that shares few words with the question. This is a limitation of
  extractive answering; the Ollama/OpenAI backends are the intended fix and the
  next thing to evaluate.

## Roadmap

- [ ] Evaluate the Ollama backend on the same set (faithfulness: are cited passages really supporting the answer?)
- [ ] Hybrid retrieval: BM25 + dense embeddings with reciprocal rank fusion
- [ ] Cross-encoder re-ranking of the top-k passages
- [ ] Larger evaluation set built from real PDFs
