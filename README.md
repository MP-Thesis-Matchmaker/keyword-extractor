# keyword-extractor

Benchmark-query synthesis for the [thesis matchmaker](../backend-core). Implements the
supervisor's suggested workflow: distill keywords from titles + abstracts of supervisors'
publications with [KeyBERT](https://github.com/MaartenGr/KeyBERT), extend them into
contextualized natural-language queries, then adjust based on trial rounds of retrievals.

## How it works

```
Postgres (backend-core's DB)
  ├─ publication ── per supervisor: title+abstract ─┐
  └─ posting ────── per posting: title+description ─┤
                                                    ▼
                                       clean (prepare_text, copied
                                       from backend-core indexing)
                                                    ▼
                                       KeyBERT (all-MiniLM-L6-v2)
                                                    ▼
                                       aggregate keyphrases
                                                    ▼
                            natural-language query (LLM via OpenAI-compatible
                            endpoint, or deterministic template when no LLM)
                                                    ▼
                            data/benchmark.jsonl  (query + gold supervisors)
                                                    ▼
                            evaluate: backend-core VectorRetriever → hit@k, MRR
```

Each benchmark row records the query, the extracted keywords, the source publication/posting
ids, the gold supervisor(s) (name + CRIS UUID where known), and generation metadata — enough
to audit any row back to its source and to compare runs across parameter changes.

## Setup

```bash
cp .env.example .env      # defaults work for the local demo DB
uv sync                   # base install: no torch, tests run offline

# Start the pre-loaded database (214k publications, 695 postings, built index):
cd ../backend-core/demo && docker compose up -d && cd -
```

## Generate benchmark queries

```bash
# Default scope: 25 sampled supervisors x <=20 recent pubs, plus all eligible postings.
uv run --extra extraction keyword-extractor generate --out data/benchmark.jsonl

# Hand-picked supervisors (best label quality; sidesteps the ORCID bug, see caveats):
uv run --extra extraction keyword-extractor generate --supervisors "Doe, Jane" "Roe, Richard"

# Offline smoke run without a model download:
KEYBERT_MODEL=fake uv run keyword-extractor generate --sample 3
```

Useful flags: `--sources publications,postings`, `--sample N` (0 = all), `--min-pubs`,
`--max-pubs`, `--language eng`, `--seed`, `--top-n`, `--query-style auto|template|llm`.

LLM phrasing: set `LLM_BASE_URL` (e.g. `http://localhost:11434/v1` with backend-core's
Ollama container) — otherwise a deterministic template is used, which keeps runs
byte-reproducible.

## Evaluate (trial retrieval rounds)

```bash
uv run --extra eval --extra extraction keyword-extractor evaluate \
    --benchmark data/benchmark.jsonl --top-k 10 --out data/eval_report.jsonl
```

Prints hit@1 / hit@5 / hit@k and MRR, overall and split by publications vs postings;
`--out` keeps per-row ranks so weak queries can be found and the KeyBERT/template
parameters adjusted (the supervisor's "play around ... based on trial rounds" step).

Evaluation imports backend-core directly (`[tool.uv.sources]` path dependency, so
`../backend-core` must be checked out) and builds the **real** `VectorRetriever`
explicitly — it refuses to run when no index manifest exists rather than silently
benchmarking backend-core's canned `FakeRetriever`. Run from the repo root: both this
package and backend-core read `.env` from the current directory.

## Tests

```bash
uv run pytest        # fully offline: fake extractor, mocked LLM transport, no DB
uv run ruff check
```

## Caveats

- **Gold-label noise (ORCID conflation).** backend-core's harvester has a known bug that
  attaches ~38k publications to wrong `uzh_authors` entries. Mitigations here: sampling
  requires a resolved CRIS UUID by default, `--supervisors` allows hand-picking, and every
  row records its `source_ids` for audit. Headline numbers inherit this upstream noise.
- **Deliberately independent keyword model.** KeyBERT runs `all-MiniLM-L6-v2`, not the
  retrieval model (`BAAI/bge-m3`) — extracting keywords in the system-under-test's own
  embedding space would measure the retriever's agreement with itself. A bge-m3 run
  (`KEYBERT_MODEL=BAAI/bge-m3`) is an upper-bound sensitivity check, not the benchmark.
- **Not a generalization test.** Queries are derived from the same corpus retrieval
  searches; this benchmarks retrieval quality, not out-of-distribution behaviour.
- **Personal data.** The DB and the generated files contain researchers' names. Supervisor
  emails are stripped at fetch time and never written; `data/` is gitignored and the
  benchmark bundle stays within the project team.
- macOS: keep `EMBEDDING_DEVICE=cpu` in `.env` for the eval step (mps aborts without a
  traceback when memory is short).
