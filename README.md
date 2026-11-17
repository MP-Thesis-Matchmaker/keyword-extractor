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
                            evaluate: POST /v1/match (themis-matcher) → hit@k, recall@k, MRR
```

Each benchmark row records the query, the extracted keywords, the source publication/posting
ids, the gold supervisor(s) (name + CRIS UUID where known), and generation metadata — enough
to audit any row back to its source and to compare runs across parameter changes. Supervisor
candidates are grouped by CRIS UUID before sampling so split name variants in DSpace (e.g.
`"Sennrich, R"` and `"Sennrich, Rico"`) have their publication counts and aliases unified.

**Complexity levels.** Every supervisor and posting yields **three rows** — `low`, `medium`,
`high` — from a single keyword extraction, so a run's row count is three times its source
count (100 supervisors → 300 rows). Both sources are sampled and both defaults are 25, so a
bare `generate` produces 150 rows; `--posting-sample 0` takes the whole corpus (579 postings
→ 1737 rows, hours under `--query-style llm`). The levels differ in kind, not just length: `low` takes
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

`evaluate` additionally needs the matcher's HTTP API running. It is a separate process and
a separate install — nothing from backend-core is imported here:

```bash
cd ../backend-core
uv sync --package themis-matcher --extra embeddings
uv run themis-matcher serve          # 127.0.0.1:8100 by default
```

## Generate benchmark queries

```bash
# Default scope: 25 sampled supervisors x <=20 recent pubs, plus 25 sampled postings.
uv run --extra extraction keyword-extractor generate --out data/benchmark.jsonl

# Generate the 900-row v4 baseline (150 supervisors + 150 postings, 3 complexities each):
uv run --extra extraction keyword-extractor generate \
    --supervisor-sample 150 --posting-sample 150 \
    --difficulty-threshold 10 --seed 42 \
    --out data/benchmark_v4.jsonl

# Hand-picked supervisors (best label quality; sidesteps the ORCID bug, see caveats):
uv run --extra extraction keyword-extractor generate \
    --supervisor-names "Doe, Jane" "Roe, Richard"

# Offline smoke run without a model download:
KEYBERT_MODEL=fake uv run keyword-extractor generate --supervisor-sample 3
```

**Flags name the entity they scope**, because the earlier `--sample` / `--min-pubs` /
`--max-pubs` read as if they controlled the same thing while controlling three different
axes. The old spellings now exit with the name of their replacement.

| flag | scopes |
|---|---|
| `--supervisor-sample N` | supervisors sampled; `0` = all |
| `--supervisor-min-pubs N` | publications a supervisor needs to qualify as a candidate |
| `--supervisor-max-pubs N` | most recent publications fed to KeyBERT per supervisor |
| `--posting-sample N` | postings sampled; `0` = all 579 |
| `--publication-language` | ISO 639-3 filter; publications only — postings carry no label |
| `--supervisor-names …` | explicit names, skips sampling |
| `--keyword-top-n N` | keyphrases kept per source (overrides `KEYBERT_TOP_N`) |
| `--difficulty-threshold N` | tags supervisors with `< N` publications as `hard`, else `easy` |

`--seed` drives both samples. Also useful: `--sources publications,postings`,
`--query-style auto|template|llm`, `--require-cris-uuid` / `--no-require-cris-uuid`,
`--log-level`.

**Progress.** Long runs print a row counter with an ETA to stderr every 25 rows, so an LLM
run over thousands of sequential calls reads as working rather than hung:

```
supervisor candidates: 1871, cris-typed: 1871, sampled: 25
publications 75/75 rows
postings     25/1737 rows  eta 2h18m
```

LLM phrasing: set `LLM_BASE_URL` (e.g. `http://localhost:11434/v1` with backend-core's
Ollama container, or an OpenAI-compatible API) — otherwise a deterministic template is used,
which keeps runs byte-reproducible. Set `LLM_REASONING_EFFORT=low` for reasoning models on
large runs.

Two different failures are survived, and they are not the same problem:

- **The payload is wrong** (400/422). Optional fields the endpoint rejects are learned once
  per process and dropped — `temperature: 0` is refused outright by gpt-5 and o-series, so it
  is sent, blamed, and omitted from every later request rather than failing every row.
  Correcting the payload costs no attempt.
- **The endpoint is momentarily unable** (429, 5xx, timeout). Retried up to `LLM_MAX_ATTEMPTS`
  times with exponential backoff, honouring `Retry-After`. A run of thousands of sequential
  calls will meet a blip, and without this one blip permanently downgrades a row to template
  phrasing.

Run at `--log-level info` (the default) to see both as they happen; `httpx`'s own per-request
lines stay suppressed so the progress counter survives.

## Evaluation Ablations

Beyond standard queries, the generator supports two targeted ablation modes for the thesis.

### 1. Specificity Ablation
Tests if the embedding "smooths over" specific jargon by comparing a supervisor's *true* specialization (KeyBERT's top-ranked keyphrase) against a human-provided broad *general* topic and a related but *adjacent* subfield they do not work in. `data/specificity_v4.jsonl` evaluates 8 supervisors across all 3 rungs (24 rows):

```bash
uv run --extra extraction keyword-extractor generate \
    --specificity --difficulty-threshold 10 --seed 42 \
    --supervisor-names "Sennrich, Rico" "Aepli, Noëmi" \
    --adjacent "Sennrich, Rico=text summarisation" \
    --adjacent "Aepli, Noëmi=dialectology" \
    --general "Sennrich, Rico=natural language processing" \
    --general "Aepli, Noëmi=natural language processing" \
    --out data/specificity_v4.jsonl

# Evaluate and check whether 'true' ranks strictly better than 'general' and 'adjacent':
uv run keyword-extractor evaluate \
    --benchmark data/specificity_v4.jsonl --top-k 10 --out data/specificity_out_v4.jsonl
uv run python scripts/analyze_specificity.py data/specificity_out_v4.jsonl
```

### 2. Student Background Ablation
Tests if adding conversational student context (study program derived from department/faculty, plus an LLM- or template-generated background sentence cached per topic and seed) dilutes the core research topic. Generates 4 medium-complexity query variants per supervisor: `topic_only`, `plus_program`, `plus_background`, and `full` (`data/benchmark_ablation_v4.jsonl` covers 100 supervisors = 400 rows):

```bash
uv run --extra extraction keyword-extractor generate \
    --ablation-background \
    --supervisor-sample 100 --difficulty-threshold 10 \
    --seed 42 \
    --out data/benchmark_ablation_v4.jsonl

uv run keyword-extractor evaluate \
    --benchmark data/benchmark_ablation_v4.jsonl --top-k 10 --out data/eval_ablation_v4.jsonl

# Per-arm metrics and deltas vs topic_only:
uv run python scripts/analyze_ablation.py data/eval_ablation_v4.jsonl

# Paired significance tests (exact McNemar on Hit@1/Hit@5, sign-flip + bootstrap 95% CI on MRR):
uv run python scripts/analyze_ablation_significance.py data/eval_ablation_v4.jsonl
```

## Evaluate (trial retrieval rounds)

```bash
uv run keyword-extractor evaluate \
    --benchmark data/benchmark_v4.jsonl --top-k 10 --out data/eval_report_v4.jsonl

# Print summary tables by source type, complexity level, and difficulty tier:
uv run python scripts/analyze_eval.py data/eval_report_v4.jsonl
```

Prints hit@k, recall@k, and MRR, overall and split by publications vs postings, by complexity level, and by publication-count difficulty tier (`easy` ≥ 10 pubs vs `hard` < 10 pubs);
`--out` keeps per-row ranks so weak queries can be found and the KeyBERT/template
parameters adjusted (the supervisor's "play around ... based on trial rounds" step). Gold
matching normalises supervisor names into casefolded token sets so `"Sennrich, Rico"` and
`"Rico Sennrich"` match.

No extras: evaluation runs off the base install. It used to import backend-core in-process
through a uv path dependency, which meant a sibling checkout, a psycopg pool and a 2.27 GB
embedding model loaded here just to score a benchmark. Retrieval is an HTTP call to the
matcher now, which owns all three.

Before the first query it preflights `GET /v1/index/status`, prints which index is about
to answer — model, dimension, document count — and checks it against the benchmark's
`*.manifest.json` sidecar (`fingerprint_check` in the report):
- **Hard mismatch (abort):** the live index's `embedding_model` or `embedding_dim` differs
  from the manifest. Pass `--allow-fingerprint-mismatch` to downgrade the abort to a warning
  for intentional cross-index sensitivity runs.
- **Soft drift (warn):** the live index's `document_count` differs from the manifest's corpus
  count by more than 1%, or the benchmark has no manifest sidecar.

A matcher with no index fails the run in a second rather than after one timeout per row, with
a message naming the command that builds one. `--matcher-base-url` points a single run at a
differently configured matcher; `--label` records what that configuration was, since the
status endpoint reports the index but not the server's retrieval settings.

`--concurrency` parallelises the queries and is **clamped to 4** — but leave it at 1. The
documented ceiling is the matcher's connection pool (`max_size=5`, synchronous routes), and
the real one is lower: uvicorn dispatches those sync routes to a threadpool where they share
a single `SentenceTransformer`, whose HuggingFace fast tokenizer is Rust-backed and not
thread-safe. Two concurrent queries reconfiguring its truncation race, and one gets
`RuntimeError: Already borrowed`, which the API returns as a 500 (observed against
`themis-matcher serve` at concurrency 4). That is an upstream bug, so the flag stays, warns
above 1, and defaults to 1.

## Tests

```bash
uv run pytest        # fully offline: fake extractor, mocked LLM transport, no DB
uv run ruff check
```

## Caveats

- **Gold labels are tied to an attribution regime.** backend-core once admitted any DSpace
  authority into `uzh_authors`, conflating CRIS Person UUIDs with bare ORCIDs and making
  co-authors at other universities look supervisor-eligible. That is fixed upstream
  (`uzh_authors` now means "registered UZH researcher"), and the correction was large:
  publications with a UZH author fell 91,729 → 53,545, and distinct names 58,218 → 2,942.
  **It also invalidated the first benchmark**: 87 of `benchmark_v2.jsonl`'s 105 gold
  supervisors are no longer supervisor-eligible at all, so they cannot be retrieved and every
  score computed from that file understates the system. `benchmark_v2.jsonl` is kept under
  `data/sample/` as the pre-fix artefact. `benchmark_v3.jsonl` (also in `data/sample/`)
  suffered from a split-identity bug where 24 researchers had fragmented CRIS name records
  (e.g. `"Sennrich, R"` with 14 publications and `"Sennrich, Rico"` with 102); candidates are
  now unified by `cris_uuid` with merged publication counts and aliases, and
  `data/benchmark_v4.jsonl` is the current, unified set.

  Nothing in v2 revealed this, because a row recorded the extractor and nothing about the
  corpus. Every generate run now writes a `*.manifest.json` sidecar carrying the database's
  schema fingerprint, the corpus counts, the live index metadata, and the run's parameters,
  which `evaluate` verifies before scoring. `--require-cris-uuid` (on by default) drops
  nothing today; the candidate/cris-typed pair printed by `generate` is the invariant that
  would show it moving again.
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
  emails are stripped at fetch time and never written; ad-hoc outputs under `data/` are
  gitignored (while the reference `v4` benchmark/ablation datasets and `data/sample/` are
  tracked in the repository).
- macOS: the mps-aborts-without-a-traceback problem belongs to whoever runs
  `themis-matcher serve` (set `MATCHER_EMBEDDING_DEVICE=cpu` there). This process never loads
  an embedding model for retrieval.
