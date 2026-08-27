"""HTTP client for the matcher's retrieval API (`themis-matcher serve`).

Evaluation used to import backend-core in-process, through a uv path dependency
on a distribution that no longer exists: backend-core is a workspace now, and
its matching engine lives in `projects/matcher` behind an HTTP API. Calling that
API instead is not merely the available option, it is the better one -- scoring
a benchmark no longer needs a sibling checkout, a psycopg pool, or a 2.27 GB
embedding model loaded here to embed queries the matcher is about to embed
anyway.

Modelled on backend-core's own client of the same API, `themis_gateway.service`,
including the rule that matters most: **a refusal is identified by its `code`,
never by the wording of its message.** The message is for humans and may be
reworded upstream at any time.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from keyword_extractor.config import Settings, get_settings

# Codes the matcher answers with, from backend-core's api/app.py. Kept as
# constants so a typo is a NameError here rather than a branch that never fires.
INDEX_NOT_BUILT = "index_not_built"
INDEX_RUN_IN_PROGRESS = "index_run_in_progress"
MODEL_MISMATCH = "model_mismatch"
DATABASE_UNAVAILABLE = "database_unavailable"
NO_SUCH_RUN = "no_such_run"


class MatcherError(RuntimeError):
    """The matcher could not be reached, or answered in a way we cannot use."""


class IndexNotBuiltError(MatcherError):
    """The matcher is running but has nothing indexed to retrieve from."""


class MatcherUnavailableError(MatcherError):
    """The matcher is unreachable, saturated, or refused in a way we cannot fix."""


class MatcherClient:
    """One session against one matcher.

    Holds a single `httpx.Client` so a run of hundreds of queries reuses the
    connection instead of completing a TCP and TLS handshake per row. `client` is
    injectable so tests drive the whole class over `httpx.MockTransport` without
    a server.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        retry_backoff: float = 1.0,
    ) -> None:
        settings = settings or get_settings()
        self._base_url = (base_url or settings.matcher_base_url).rstrip("/")
        self._max_attempts = max(1, settings.matcher_max_attempts)
        self._retry_backoff = retry_backoff
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.matcher_timeout_s,
                write=settings.matcher_timeout_s,
                pool=settings.matcher_timeout_s,
            )
        )

    def __enter__(self) -> MatcherClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Close the underlying client, unless it was handed to us."""
        if self._owns_client:
            self._client.close()

    @property
    def base_url(self) -> str:
        return self._base_url

    def index_status(self) -> dict[str, Any]:
        """What the matcher has indexed: model, dimension, document count, sources.

        The preflight for any evaluation run. An eval report that does not say
        which index answered is not reproducible, and a run against a matcher
        with no index should fail in a second rather than after N timeouts.
        """
        return self._request("GET", "/v1/index/status")

    def match(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Ranked supervisors for one query.

        Returns the raw `SupervisorMatch` dicts rather than a parsed model: the
        evaluator's `FindFn` seam is defined in terms of dicts, and the extra
        fields the matcher reports (`has_uzh_affiliation`, `evidence`,
        `publication_count`) are worth passing through untouched to whatever
        writes the per-row output.
        """
        body = self._request("POST", "/v1/match", json={"query": query, "top_k": top_k})
        return body["matches"]

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, *, json: dict | None = None) -> dict[str, Any]:
        """One call, retried only for conditions that a retry can actually clear."""
        url = f"{self._base_url}{path}"
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.request(method, url, json=json)
            except httpx.HTTPError as exc:
                last = MatcherUnavailableError(f"could not reach the matcher at {url}: {exc}")
            else:
                if response.status_code < 400:
                    return response.json()
                code, message = _refusal(response)
                # A 409 is a standing condition -- no index, a run in flight, a
                # model mismatch. Retrying it just spends the run's time
                # rediscovering the same answer.
                if code != DATABASE_UNAVAILABLE:
                    raise _error_for(code, message, response, url)
                last = MatcherUnavailableError(_database_unavailable_message(message, url))
            if attempt < self._max_attempts:
                time.sleep(self._retry_backoff * 2 ** (attempt - 1))
        raise last if last else MatcherUnavailableError(f"the matcher at {url} did not answer")


def _refusal(response: httpx.Response) -> tuple[str, str]:
    """The matcher's `(code, message)`, tolerating a body that is not its ApiError."""
    try:
        error = response.json()
    except ValueError:
        return "", response.text
    if not isinstance(error, dict):
        return "", response.text
    return error.get("code", ""), error.get("message") or response.text


def _error_for(code: str, message: str, response: httpx.Response, url: str) -> MatcherError:
    """Map a refusal onto a message that says what to do about it."""
    if code == INDEX_NOT_BUILT:
        return IndexNotBuiltError(
            f"the matcher at {url} has no index, so there is nothing to retrieve from "
            f"({message}). Build one with `themis-matcher index --source db` in "
            "backend-core, or point --matcher-base-url at a matcher that has one."
        )
    if code == INDEX_RUN_IN_PROGRESS:
        return MatcherUnavailableError(
            f"the matcher at {url} is indexing right now ({message}). Wait for it to "
            "finish -- `GET /v1/index/runs` reports progress -- and re-run."
        )
    if code == MODEL_MISMATCH:
        return MatcherUnavailableError(
            f"the matcher at {url} is configured for a different embedding model than "
            f"the one its index was built with ({message}). Align "
            "MATCHER_EMBEDDING_MODEL with the index, or rebuild the index."
        )
    return MatcherUnavailableError(
        f"the matcher at {url} refused the request "
        f"({response.status_code} {code or 'unknown'}): {message}"
    )


def _database_unavailable_message(message: str, url: str) -> str:
    return (
        f"the matcher at {url} could not reach its database, or its connection pool is "
        f"saturated ({message}). The pool caps at 5 and every route is synchronous, so "
        "if this appears under load, lower --concurrency."
    )
