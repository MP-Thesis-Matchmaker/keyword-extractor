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
                            KeyBERT (paraphrase-multilingual-MiniLM-L12-v2)
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

**Complexity levels.** Every supervisor and posting yields **three rows** — `low`, `medium`,
`high` — from a single keyword extraction, so a run's row count is three times its source
count (100 supervisors → 300 rows). The levels differ in kind, not just length: `low` takes
the 2 strongest keywords in one short sentence, `medium` takes 4, and `high` takes 8 in a
multi-sentence query that names the subject area and asks to combine the topics. Evaluation
reports hit@k and MRR per level, which is the point of generating them: it measures what a
vaguer query costs. The full keyword list is kept in `metadata.all_keywords` regardless of
level.

**`query_style` is what happened, not what was asked.** A row produced by the LLM records
`"llm"`; one where the endpoint failed and the template took over records `"template"`, and
the run prints a warning naming how many rows fell back. A benchmark that silently claims
LLM phrasing for template text is worse than one that admits the fallback.

**Languages.** The corpus is not English-only: 5,956 of the eligible publications are German
(plus French, Italian, Spanish tails), and thesis postings carry no language label at all in
the database. One multilingual keyword model therefore handles everything, and each row's
language is *detected from the text* it was extracted from (`metadata.text_language`) rather
than taken from `publication.language` (`metadata.db_language`, recorded alongside for
comparison). German rows get German query phrasing; the smaller French/Italian/Spanish tail
keeps native keywords inside an English carrier sentence.

## Setup

> `.env` is gitignored, so it does **not** pick up changes to `.env.example` — and an entry
> there silently overrides a changed default (a stale `KEYBERT_MODEL` once kept a run on the
> old English-only model). Diff the two after pulling.

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
`--max-pubs`, `--seed`, `--top-n`, `--query-style auto|template|llm`, and `--language` to
restrict publications to one ISO 639-3 code (default: all languages).

LLM phrasing: set `LLM_BASE_URL` (e.g. `http://localhost:11434/v1` with backend-core's
Ollama container, or an OpenAI-compatible API) — otherwise a deterministic template is used,
which keeps runs byte-reproducible. Optional request fields the endpoint rejects are learned
once per process and dropped: `temperature: 0` is refused outright by gpt-5 and o-series
models, so it is sent, blamed, and omitted from every later request rather than failing every
row. Set `LLM_REASONING_EFFORT=low` for reasoning models on large runs.

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
- **Deliberately independent keyword model.** KeyBERT runs
  `paraphrase-multilingual-MiniLM-L12-v2`, not the retrieval model (`BAAI/bge-m3`) —
  extracting keywords in the system-under-test's own embedding space would measure the
  retriever's agreement with itself. A bge-m3 run (`KEYBERT_MODEL=BAAI/bge-m3`) is an
  upper-bound sensitivity check, not the benchmark.
- **`publication.language` describes the record, not the abstract.** In a 300-row sample of
  `deu`-labelled publications, 87 carried a fully English abstract (German title, English
  abstract). That is why query language comes from detection rather than the column, and why
  both values are recorded per row.
- **Truncated document view.** The multilingual model ships `max_seq_length` 128 against a
  median abstract of roughly 240 tokens, so keyphrases are ranked against the document's
  opening rather than all of it. Candidate phrases themselves come from the full text, and
  supervisor rows pool up to 20 publications, which absorbs most of this. Set
  `KEYBERT_MAX_SEQ_LENGTH=256` for a sensitivity run.
- **Language detection is a heuristic** (distinctive function-word overlap, no extra
  dependency). A misfire changes only the query phrasing, never a gold label.
- **Bigrams can be words that were never adjacent.** KeyBERT builds candidates with
  scikit-learn's `CountVectorizer`, which removes stop words *before* forming n-grams, so
  `keyphrase_ngram_range=(1, 2)` can emit pairs like `kriminalistischer juristischer` whose
  words sit one or two tokens apart in the abstract (verified: none of the sampled bigrams
  occur adjacently in their source text). Affects both languages and predates the
  multilingual switch. Fix if it hurts retrieval: `KEYBERT_NGRAM_MAX=1` for single words.
- **Not a generalization test.** Queries are derived from the same corpus retrieval
  searches; this benchmarks retrieval quality, not out-of-distribution behaviour.
- **Personal data.** The DB and the generated files contain researchers' names. Supervisor
  emails are stripped at fetch time and never written; `data/` is gitignored and the
  benchmark bundle stays within the project team.
- macOS: keep `EMBEDDING_DEVICE=cpu` in `.env` for the eval step (mps aborts without a
  traceback when memory is short).
