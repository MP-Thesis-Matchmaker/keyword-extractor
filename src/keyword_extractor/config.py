"""Application settings, loaded from environment variables and an optional .env file.

Mirrors backend-core's Settings pattern (pydantic-settings, .env read from the
CWD, unknown keys ignored). One .env still serves both repos, but for a
different reason than it used to: no backend-core code runs in this process any
more, so nothing here is being configured on its behalf. What is shared now are
the variables that name the same things on both sides -- DATABASE_URL, and
MATCHER_BASE_URL, which backend-core deliberately exempts from its per-member
`env_prefix` so every member and every client of the matcher spells it the same
way.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config for the benchmark-query synthesiser."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Same Postgres backend-core writes; this repo only reads from it.
    database_url: str = "postgresql://matchmaker:matchmaker@localhost:5432/matchmaker"

    # KeyBERT embedding model. The special value "fake" selects the deterministic
    # offline extractor (tests, CI, no torch).
    #
    # Multilingual because the corpus is: about 10% of publications with an
    # abstract and a UZH author are not English (5,956 deu, plus fra/ita/spa
    # tails), and `posting.language` is NULL for every posting, so there is no
    # column to route German documents to a second model with even if we wanted
    # to. This is the model KeyBERT's README recommends for that case.
    #
    # Still deliberately NOT bge-m3, the retrieval model: picking keywords in the
    # embedding space of the system under test would measure the retriever's
    # agreement with itself.
    keybert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    keybert_top_n: int = 8

    # Token cap for the document embedding KeyBERT ranks candidates against.
    # None keeps whatever the model ships -- 128 for the multilingual MiniLM,
    # which is what it was distilled at. Raising it to 256 is architecturally
    # fine (the checkpoint carries 512 position embeddings) but is a deliberate
    # sensitivity run rather than a default, since those positions were never
    # tuned during the paraphrase distillation.
    #
    # The truncation matters less than it looks: candidate n-grams come from the
    # FULL text through KeyBERT's CountVectorizer, only the document vector is
    # capped, and a supervisor row pools keyphrases over up to 20 publications.
    keybert_max_seq_length: int | None = None
    # keyphrase_ngram_range = (1, keybert_ngram_max)
    keybert_ngram_max: int = 2
    # Maximal Marginal Relevance: trades raw similarity for diversity, so the
    # keyword set spans a supervisor's topics instead of restating one phrase.
    keybert_use_mmr: bool = True
    keybert_diversity: float = 0.5

    # OpenAI-compatible chat endpoint for query synthesis. Unset means the
    # deterministic template synthesiser is used and no LLM is needed --
    # the same toggle backend-core uses for its query parser.
    llm_base_url: str | None = None
    llm_model: str = "llama3.1"
    llm_api_key: str | None = None

    # Only for reasoning models (gpt-5, o-series, Qwen3, DeepSeek-R1 and
    # friends): "minimal"/"low"/"medium"/"high" bound the hidden reasoning an
    # answer is allowed to cost. Left unset because the field is meaningless to
    # a non-reasoning model and some endpoints reject fields they do not know --
    # synthesis/llm.py drops it and retries when that happens. Worth setting for
    # a large run: a reasoning model can think for many seconds per query.
    llm_reasoning_effort: str | None = None

    # Attempts per query when the endpoint fails *transiently* -- a 429, a 5xx,
    # a read timeout. Unrelated to the unsupported-field retry in synthesis/llm.py,
    # which corrects the payload and costs no attempt. Worth having because a
    # long run makes thousands of sequential calls and a single blip would
    # otherwise downgrade that row to template phrasing for good.
    llm_max_attempts: int = 3

    # The matcher's HTTP API (`themis-matcher serve`). Retrieval is a network
    # call now: evaluation used to import backend-core in-process, which meant a
    # sibling checkout, a psycopg pool and a 2.27 GB embedding model loaded into
    # this process to score a benchmark. The matcher owns all three.
    #
    # Default matches `themis-matcher serve`'s own MATCHER_API_HOST/PORT
    # defaults, so a local run needs no configuration at all. Unlike the gateway,
    # which fails when this is unset because it may legitimately run without a
    # matcher, a benchmark run has nothing to do without one.
    matcher_base_url: str = "http://127.0.0.1:8100"

    # Long enough for a cold query -- the matcher may still be loading bge-m3 on
    # its first request. Copied from the gateway's DEFAULT_TIMEOUT_S.
    matcher_timeout_s: float = 60.0

    # Attempts per query when the matcher fails *transiently* -- a saturated
    # connection pool, a read timeout, a dropped connection. Never applied to a
    # 409: those are standing conditions (no index, a run in flight, a model
    # mismatch) that retrying cannot clear.
    matcher_max_attempts: int = 3

    # Concurrent /v1/match calls. Left at 1, and not for the documented reason:
    # the matcher's psycopg pool caps at 5, but well before that ceiling its
    # query embedding does. Uvicorn runs the matcher's synchronous routes in a
    # threadpool, and they share one SentenceTransformer whose HuggingFace fast
    # tokenizer is Rust-backed and not thread-safe -- two concurrent queries race
    # on its truncation settings and one gets `RuntimeError: Already borrowed`,
    # surfacing as a 500. Raise this only against a matcher that has fixed that.
    matcher_concurrency: int = 1


def get_settings() -> Settings:
    """Return settings, read fresh from the environment."""
    return Settings()
