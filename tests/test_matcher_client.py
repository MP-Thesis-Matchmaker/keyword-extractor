"""The matcher HTTP client -- entirely offline, over httpx.MockTransport."""

import httpx
import pytest

from keyword_extractor.matcher_client import (
    IndexNotBuiltError,
    MatcherClient,
    MatcherUnavailableError,
)

_MATCHES = [
    {"supervisor": "Doe, Jane", "score": 0.81, "has_uzh_affiliation": True},
    {"supervisor": "Roe, Richard", "score": 0.64, "has_uzh_affiliation": False},
]

_STATUS = {
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": 1024,
    "document_count": 215451,
    "sources": "postgres",
}


def _client(handler, **kwargs):
    return MatcherClient(
        base_url="http://matcher.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_backoff=0.0,
        **kwargs,
    )


def _refusal(status, code, message="refused"):
    return httpx.Response(status, json={"code": code, "message": message})


def test_match_posts_the_query_and_returns_the_matches():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"matches": _MATCHES})

    assert _client(handler).match("dense retrieval", top_k=7) == _MATCHES
    assert seen["url"] == "http://matcher.test/v1/match"
    assert seen["method"] == "POST"
    assert seen["body"] == {"query": "dense retrieval", "top_k": 7}


def test_index_status_reads_the_manifest():
    status = _client(lambda request: httpx.Response(200, json=_STATUS)).index_status()
    assert status["embedding_model"] == "BAAI/bge-m3"
    assert status["document_count"] == 215451


def test_index_not_built_names_the_remediation():
    client = _client(lambda request: _refusal(409, "index_not_built"))
    with pytest.raises(IndexNotBuiltError) as excinfo:
        client.match("anything")
    assert "themis-matcher index" in str(excinfo.value)


def test_index_not_built_is_recognised_by_code_not_wording():
    """The message is for humans and may be reworded upstream at any time."""
    client = _client(lambda request: _refusal(409, "index_not_built", "totally different phrasing"))
    with pytest.raises(IndexNotBuiltError):
        client.match("anything")


def test_unknown_refusal_still_raises_rather_than_returning_junk():
    client = _client(lambda request: httpx.Response(500, text="<html>gateway exploded</html>"))
    with pytest.raises(MatcherUnavailableError) as excinfo:
        client.match("anything")
    assert "500" in str(excinfo.value)


def test_database_unavailable_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setenv("MATCHER_MAX_ATTEMPTS", "3")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return _refusal(503, "database_unavailable", "pool timeout")
        return httpx.Response(200, json={"matches": _MATCHES})

    assert _client(handler).match("q") == _MATCHES
    assert len(calls) == 3


def test_database_unavailable_exhausted_names_the_pool_ceiling(monkeypatch):
    monkeypatch.setenv("MATCHER_MAX_ATTEMPTS", "2")
    calls = []

    def handler(request):
        calls.append(request)
        return _refusal(503, "database_unavailable", "pool timeout")

    with pytest.raises(MatcherUnavailableError) as excinfo:
        _client(handler).match("q")
    assert len(calls) == 2
    assert "--concurrency" in str(excinfo.value)


def test_model_mismatch_is_not_retried(monkeypatch):
    """A 409 is a standing condition; retrying only spends the run's time."""
    monkeypatch.setenv("MATCHER_MAX_ATTEMPTS", "5")
    calls = []

    def handler(request):
        calls.append(request)
        return _refusal(409, "model_mismatch", "index built with X, server configured for Y")

    with pytest.raises(MatcherUnavailableError):
        _client(handler).match("q")
    assert len(calls) == 1


def test_index_run_in_progress_is_not_retried(monkeypatch):
    monkeypatch.setenv("MATCHER_MAX_ATTEMPTS", "5")
    calls = []

    def handler(request):
        calls.append(request)
        return _refusal(409, "index_run_in_progress", "run 12 is running")

    with pytest.raises(MatcherUnavailableError) as excinfo:
        _client(handler).match("q")
    assert len(calls) == 1
    assert "index/runs" in str(excinfo.value)


def test_transport_error_is_retried_then_reported(monkeypatch):
    monkeypatch.setenv("MATCHER_MAX_ATTEMPTS", "3")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(MatcherUnavailableError) as excinfo:
        _client(handler).match("q")
    assert len(calls) == 3
    assert "could not reach the matcher" in str(excinfo.value)
