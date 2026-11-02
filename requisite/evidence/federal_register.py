"""The Federal Register's public API, read-only (decisions, "Public sources are reached through their official APIs"):
a final rule's effective date, by the Federal Register document number regulations.gov lists it under (frDocNum).

regulations.gov states no effective date for a rule, in its documents listing or in the rule's own record (its
effectiveDate is null in both: checked November 2 on AMS_FRDOC_0001-2765 and APHIS-2019-0024-0025); the Federal
Register publishes it as effective_on, with the rule's DATES text. Whether a measure is "final but not yet in effect"
(the objective's events) is read from there. Every response is recorded (cassettes) and committed by its caller as an
artifact, so a decision that rests on it recomputes from what was retrieved. Several documents are asked in one request,
as the API allows; a number it does not find is its answer (errors.not_found), kept.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from requisite import cassettes
from requisite.evidence.fetched import USER_AGENT, Fetched, fetched, free_budget, kept

SOURCE = "federalregister.gov"
PROVIDER = "federal_register"
API = "https://www.federalregister.gov/api/v1/documents/"
FIELDS = ("document_number", "type", "publication_date", "effective_on", "dates")
BATCH = 20  # document numbers per request
TIMEOUT = 60.0
RETRIES = 5
TRANSIENT = frozenset({429, 500, 502, 503, 504})


class FederalRegisterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Published:
    """What the Federal Register states about one document: found or not; its type; its effective date (None where it
    states none); its DATES text as published."""

    number: str
    found: bool
    type: str | None
    effective_on: date | None
    dates: str | None


@dataclass
class FederalRegister:
    transport: httpx.BaseTransport | None = None
    root: Path | None = None
    budget: cassettes.Budget = field(default_factory=free_budget)

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            transport=self.transport, timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def _send(self, url: str) -> dict[str, Any]:
        for attempt in range(1, RETRIES + 2):
            try:
                response = self._http.get(url, params=[("fields[]", f) for f in FIELDS])
            except httpx.TransportError as e:
                if attempt > RETRIES:
                    raise FederalRegisterError(f"{url}: no response after {RETRIES} retries: {e!r}") from e
                time.sleep(2**attempt)
                continue
            if response.status_code == 200:
                return kept(response, ("content-type",))
            if response.status_code in TRANSIENT and attempt <= RETRIES:
                time.sleep(2**attempt)
                continue
            raise FederalRegisterError(f"{url}: HTTP {response.status_code}: {response.text[:300]}")
        raise AssertionError("unreachable")

    def documents(self, numbers: Iterable[str]) -> tuple[list[Fetched], dict[str, Published]]:
        """What the Federal Register states about each document number, and the responses it rests on."""
        wanted = sorted(set(numbers))
        responses: list[Fetched] = []
        out: dict[str, Published] = {}
        for i in range(0, len(wanted), BATCH):
            batch = wanted[i : i + BATCH]
            path = ",".join(batch) + ".json"
            url = API + path

            def live(url: str = url) -> tuple[Any, float]:
                return self._send(url), 0.0

            recorded = cassettes.call(PROVIDER, f"documents/{path}", {"fields": list(FIELDS)}, live, budget=self.budget,
                                      estimate_usd=0.0, root=self.root)  # fmt: skip
            response = fetched(SOURCE, f"{API}{path}?" + "&".join(f"fields[]={f}" for f in FIELDS), recorded)
            responses.append(response)
            out.update(_published(batch, response.json()))
        return responses, out

    def close(self) -> None:
        self._http.close()


def asked_in(locator: str) -> list[str]:
    """The document numbers a response was asked about, read from its locator."""
    return locator.split("?")[0].removeprefix(API).removesuffix(".json").split(",")


def stated(locator: str, content: bytes) -> dict[str, Published]:
    """What a recorded response states about each number it was asked about."""
    return _published(asked_in(locator), json.loads(content))


def _published(asked: list[str], body: Mapping[str, Any]) -> dict[str, Published]:
    """A response's documents: a list under results when several were asked, the document itself when one was."""
    results = body.get("results") if "results" in body or "count" in body else [body]
    found = {str(r["document_number"]): r for r in results or []}
    out = {}
    for n in asked:
        r = found.get(n)
        if r is None:
            out[n] = Published(number=n, found=False, type=None, effective_on=None, dates=None)
            continue
        eff = r.get("effective_on")
        out[n] = Published(number=n, found=True, type=r.get("type"), effective_on=date.fromisoformat(eff) if eff else None,
                           dates=r.get("dates"))  # fmt: skip
    return out
