"""Test isolation from the developer's local configuration.

`Settings` reads `.env` from the current working directory, and environment
variables take priority over it. Without this fixture a test that builds a
synthesiser through `--query-style auto` picks up a real `LLM_BASE_URL` and
`LLM_API_KEY` from the developer's own `.env` and makes live, billed API calls;
`monkeypatch.delenv` does not prevent that, because the value comes from the
file rather than the environment.

Autouse and unconditional: the suite must be runnable offline, and no test may
ever reach a real endpoint or download a model by accident.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    # Empty string, not delenv: an env var set to "" overrides the .env file and
    # is falsy, so `build_synthesizer` takes the offline template path.
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    # No model download in the suite; individual tests may still override.
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:1/nonexistent_test")
    # One attempt: a test that deliberately points at an unreachable host must
    # not spend the retry backoff. Retry behaviour is tested directly, with an
    # injected transport and zero backoff.
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
