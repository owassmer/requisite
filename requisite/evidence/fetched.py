"""A response from a public source as it was returned, shared by every read-only adapter (regulations.gov, Kalshi).

A response is kept as the bytes the source returned (text where they are UTF-8, base64 otherwise), with its status and
the headers that describe it, so the artifact a replay commits is the artifact the live run committed (spec I2). Every
request names Requisite by one User-Agent (decisions, "Public sources are reached through their official APIs").
"""

from __future__ import annotations

import base64
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from requisite import cassettes

# decisions, "Public sources are reached through their official APIs": the conventional self-identifying crawler form,
# which regulations.gov's file CDN requires.
USER_AGENT = "Mozilla/5.0 (compatible; RequisiteResearch/1.0; +https://github.com/owassmer/requisite)"


@dataclass(frozen=True)
class Fetched:
    """One response as the source returned it: the source it came from, its locator (the URL without any key), status,
    media type, bytes, and when it was retrieved, as its cassette recorded it (so a replay reproduces the same
    artifact)."""

    source: str
    locator: str
    status: int
    media_type: str
    content: bytes
    retrieved_at: datetime

    def json(self) -> Any:
        return json.loads(self.content)


def kept(response: httpx.Response, headers: Iterable[str]) -> dict[str, Any]:
    """A response as a cassette keeps it: status, the named headers it carries, and its bytes exactly."""
    content = response.content
    try:
        body: dict[str, Any] = {"text": content.decode("utf-8")}
    except UnicodeDecodeError:
        body = {"base64": base64.b64encode(content).decode("ascii")}
    named = {k: response.headers[k] for k in headers if k in response.headers}
    return {"status": response.status_code, "headers": named, **body}


def fetched(source: str, where: str, recorded: cassettes.Recorded) -> Fetched:
    """The response a cassette holds (recorded now or replayed), as the source returned it."""
    k = recorded.response
    content = k["text"].encode("utf-8") if "text" in k else base64.b64decode(k["base64"])
    return Fetched(
        source=source,
        locator=where,
        status=int(k["status"]),
        media_type=str(k["headers"].get("content-type", "application/octet-stream")),
        content=content,
        retrieved_at=datetime.fromisoformat(recorded.recorded_at),
    )


def free_budget() -> cassettes.Budget:
    """The budget of a run's requests to a public source that charges nothing: no dollar cap can bind them, and each is
    still logged as a call (spec §6.6); the source's own rate limit is the client's to respect."""
    return cassettes.Budget(run_usd=0.0, run_calls=sys.maxsize, day_usd=math.inf)
