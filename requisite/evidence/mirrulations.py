"""Mirrulations, a public mirror of regulations.gov (Moravian University; AWS Open Data, s3://mirrulations), read-only:
a comment's record as regulations.gov's API returns it, without the API's hourly limit.

Why a mirror (decisions, "Sources": amended November 2 at Owen's question). regulations.gov's API allows one key 1,000
requests an hour, and reading a sample spends one request per comment record. The mirror stores each record as the
API's detail response (checked November 2: the fixture's comment equal in every attribute and its included attachment,
and 12 of 12 comments posted November 1, 2026 equal on every field their listing entry carries; it lacks only the API's
"links" block). A mirror's copy is evidence only where it is shown current: requisite/evidence/items.py takes it only
where every field the frame's official listing gives for that comment (its lastModifiedDate among them) is equal, and
reads the API otherwise. Attachments keep coming from regulations.gov's file host, which no hourly limit binds.

Each response is recorded (cassettes, provider "mirrulations") and committed as an artifact whose source names the
mirror, so what was read from where recomputes, and a clean copy replays it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from requisite import cassettes
from requisite.evidence.fetched import USER_AGENT, Fetched, fetched, free_budget, kept

SOURCE = "mirrulations (a mirror of regulations.gov)"
PROVIDER = "mirrulations"
BUCKET = "https://mirrulations.s3.amazonaws.com/"
TIMEOUT = 60.0
RETRIES = 5
TRANSIENT = frozenset({429, 500, 502, 503, 504})


class MirrulationsError(RuntimeError):
    pass


def comment_key(comment_id: str) -> str:
    """Where the mirror keeps a comment's record: raw-data/<agency>/<docket>/text-<docket>/comments/<id>.json, the
    docket being the comment id less its last part (as regulations.gov numbers comments within their docket)."""
    docket = comment_id.rsplit("-", 1)[0]
    agency = docket.split("-", 1)[0]
    return f"raw-data/{agency}/{docket}/text-{docket}/comments/{comment_id}.json"


@dataclass
class Mirrulations:
    transport: httpx.BaseTransport | None = None
    root: Path | None = None
    budget: cassettes.Budget = field(default_factory=free_budget)

    def __post_init__(self) -> None:
        self._http = httpx.Client(transport=self.transport, timeout=TIMEOUT, follow_redirects=True,
                                  headers={"User-Agent": USER_AGENT})  # fmt: skip

    def _send(self, url: str) -> dict[str, Any]:
        for attempt in range(1, RETRIES + 2):
            try:
                response = self._http.get(url)
            except httpx.TransportError as e:
                if attempt > RETRIES:
                    raise MirrulationsError(f"{url}: no response after {RETRIES} retries: {e!r}") from e
                time.sleep(2**attempt)
                continue
            if response.status_code in (200, 403, 404):  # the bucket answers 403 for a key it does not hold
                return kept(response, ("content-type", "last-modified"))
            if response.status_code in TRANSIENT and attempt <= RETRIES:
                time.sleep(2**attempt)
                continue
            raise MirrulationsError(f"{url}: HTTP {response.status_code}: {response.text[:300]}")
        raise AssertionError("unreachable")

    def comment(self, comment_id: str) -> Fetched | None:
        """A comment's record as the mirror holds it, or None where it holds none."""
        url = BUCKET + quote(comment_key(comment_id))

        def live() -> tuple[Any, float]:
            return self._send(url), 0.0

        recorded = cassettes.call(PROVIDER, url, {}, live, budget=self.budget, estimate_usd=0.0, root=self.root)
        got = fetched(SOURCE, url, recorded)
        return got if got.status == 200 else None

    def close(self) -> None:
        self._http.close()
