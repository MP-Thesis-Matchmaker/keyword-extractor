"""Application settings, loaded from environment variables and an optional .env file.

Mirrors backend-core's Settings pattern (pydantic-settings, .env read from the
CWD, unknown keys ignored) so one .env at the repo root serves both this
package and backend-core code imported during evaluation.
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
    # offline extractor (tests, CI, no torch). Deliberately NOT bge-m3 by default:
    # picking keywords in the retrieval model's own embedding space would measure
    # the retriever's agreement with itself.
    keybert_model: str = "all-MiniLM-L6-v2"
    keybert_top_n: int = 8
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


def get_settings() -> Settings:
    """Return settings, read fresh from the environment."""
    return Settings()
