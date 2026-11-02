"""A regulations.gov v4 served offline: the recorded responses in fixtures/regulations_gov/FAA-2026-2730 (real shapes),
answered through an httpx transport with the API's documented behaviour (its OpenAPI specification and documentation):
filters, sort, at most 20 pages of 5 to 250 results, the lastModifiedDate filter read in Eastern time as
"yyyy-MM-dd HH:mm:ss", the posted-date filter by day, a rolling hourly limit stated in x-ratelimit-remaining and
enforced with 429. A test adds resources in the same shapes, and scripts failures before a path answers.

Nothing here is the product: it stands in for the network, which the sandbox and CI do not have."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

import httpx

from requisite.evidence.regulations_gov import USER_AGENT

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "regulations_gov" / "FAA-2026-2730"
EASTERN = ZoneInfo("America/New_York")
KEY = "fake-gov-key-for-tests-0123456789"
DOCKET = "FAA-2026-2730"
PDF_0005 = "https://downloads.regulations.gov/FAA-2026-2730-0005/attachment_1.pdf"
PDF_0003 = "https://downloads.regulations.gov/FAA-2026-2730-0003/attachment_1.pdf"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def fixture_pdf() -> bytes:
    return (FIXTURES / "attachment_FAA-2026-2730-0005_1.pdf").read_bytes()


def when(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class FakeTime:
    """A clock and a sleep for the pacer: sleeping advances the clock, and nothing waits in real time."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@dataclass
class FakeAPI:
    time: FakeTime = field(default_factory=FakeTime)
    limit: int = 1000
    window: float = 3600.0
    latency: float = 1.0  # seconds of fake time each request takes
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> listing resource
    comments: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)  # id -> (docket, listing resource)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)  # "comments/ID" etc. -> body
    files: dict[str, bytes] = field(default_factory=dict)
    script: dict[str, list[int]] = field(default_factory=dict)  # a path or URL -> statuses served before it answers
    retry_after: str | None = None
    unreachable: int = 0  # the next this-many requests get no response (the network down)
    after: list[Callable[[httpx.Request], None]] = field(default_factory=list)  # changes to the record, after a request
    requests: list[httpx.Request] = field(default_factory=list)
    served: list[int] = field(default_factory=list)
    _sent: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ---------------------------------------------------------------- serving

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            if self.unreachable > 0:  # the network down: no response at all, for this many requests
                self.unreachable -= 1
                self.time.now += self.latency
                raise httpx.ConnectTimeout("The handshake operation timed out", request=request)
            self.requests.append(request)
            response = self._answer(request)
            self.served.append(response.status_code)
            self.time.now += self.latency
            for change in self.after:
                change(request)
            return response

    def _answer(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == USER_AGENT
        assert "api_key" not in str(request.url) and KEY not in str(request.url)
        url = request.url
        if url.host == "downloads.regulations.gov":
            assert "x-api-key" not in request.headers, "the file host gets no key"
            where = f"https://{url.host}{url.path}"
            if (status := self._scripted(where)) is not None:
                return httpx.Response(status, text="scripted")
            if where not in self.files:
                return httpx.Response(403, text="<Error><Code>AccessDenied</Code></Error>")
            return httpx.Response(200, content=self.files[where], headers={"content-type": "application/pdf"})
        assert url.host == "api.regulations.gov" and url.path.startswith("/v4/")
        assert request.headers.get("x-api-key") == KEY
        path = url.path.removeprefix("/v4/")
        now = self.time.now
        self._sent = [t for t in self._sent if t > now - self.window]
        if len(self._sent) >= self.limit:
            return httpx.Response(429, json={"error": {"code": "OVER_RATE_LIMIT"}}, headers=self._limits(0))
        self._sent.append(now)
        headers = self._limits(self.limit - len(self._sent))
        if (status := self._scripted(path)) is not None:
            extra = {"retry-after": self.retry_after} if status == 429 and self.retry_after else {}
            return httpx.Response(status, json={"error": "scripted"}, headers={**headers, **extra})
        params = dict(parse_qsl(urlsplit(str(url)).query, keep_blank_values=True))
        if path in ("documents", "comments"):
            return self._listing(path, params, headers)
        if path in self.details:
            return httpx.Response(200, json=self.details[path], headers=headers)
        return httpx.Response(404, json={"errors": [{"status": "404", "title": "Not Found"}]}, headers=headers)

    def _scripted(self, where: str) -> int | None:
        queue = self.script.get(where)
        return queue.pop(0) if queue else None

    def _limits(self, remaining: int) -> dict[str, str]:
        return {"x-ratelimit-limit": str(self.limit), "x-ratelimit-remaining": str(remaining),
                "content-type": "application/vnd.api+json;charset=utf-8"}  # fmt: skip

    def _listing(self, path: str, params: dict[str, str], headers: dict[str, str]) -> httpx.Response:
        size, number = int(params.get("page[size]", "25")), int(params.get("page[number]", "1"))
        if not (5 <= size <= 250 and 1 <= number <= 20):
            return httpx.Response(400, json={"errors": [{"detail": "page out of range"}]}, headers=headers)
        if path == "documents":
            rows = [(r["attributes"]["docketId"], r) for r in self.documents.values()]
        else:
            rows = list(self.comments.values())
        kept = [r for docket, r in rows if self._matches(docket, r["attributes"], params)]
        for key in reversed([k for k in params.get("sort", "").split(",") if k]):
            name = key.lstrip("-")
            kept.sort(key=(lambda r, n=name: r["id"] if n == "documentId" else r["attributes"][n] or ""),
                      reverse=key.startswith("-"))  # fmt: skip
        page = kept[(number - 1) * size : number * size]
        more = number * size < len(kept)
        meta = {"hasNextPage": more, "hasPreviousPage": number > 1, "numberOfElements": len(page), "pageNumber": number,
                "pageSize": size, "totalElements": len(kept), "totalPages": math.ceil(len(kept) / size),
                "firstPage": number == 1, "lastPage": not more}  # fmt: skip
        return httpx.Response(200, json={"data": page, "meta": meta}, headers={**headers, "content-type": "application/json"})

    @staticmethod
    def _matches(docket: str, a: dict[str, Any], params: dict[str, str]) -> bool:
        posted = when(a["postedDate"]).astimezone(EASTERN).date().isoformat()
        modified = when(a["lastModifiedDate"])
        for key, value in params.items():
            if key == "filter[searchTerm]" and value.lower() not in a.get("title", "").lower():
                return False
            if key == "filter[documentType]" and a.get("documentType") != value:
                return False
            if key == "filter[agencyId]" and a.get("agencyId") not in value.split(","):
                return False
            if key == "filter[docketId]" and docket != value:
                return False
            if key == "filter[postedDate][ge]" and posted < value:
                return False
            if key == "filter[postedDate][le]" and posted > value:
                return False
            if key == "filter[lastModifiedDate][ge]":
                since = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=EASTERN)
                if modified < since:
                    return False
        return True

    # ---------------------------------------------------------------- the world

    def add_document(self, resource: dict[str, Any], *, key: str | None = None) -> None:
        """A document in the record; `key` keeps a second document the source lists under the same id."""
        self.documents[key or resource["id"]] = resource

    def add_comment(self, docket: str, resource: dict[str, Any], detail: dict[str, Any] | None = None, *,
                    key: str | None = None) -> None:  # fmt: skip
        """A listed comment; `key` lets one comment id be listed for two records, as regulations.gov can list it."""
        self.comments[key or resource["id"]] = (docket, resource)
        if detail is not None:
            self.details[f"comments/{resource['id']}"] = detail


def comment_entry(comment_id: str, posted: str, modified: str, *, withdrawn: bool = False) -> dict[str, Any]:
    """A comment as the listing serves it, in the recorded shape."""
    entry = copy.deepcopy(load("comments_list.json")["body"]["data"][0])
    entry["id"] = comment_id
    entry["attributes"].update(postedDate=posted, lastModifiedDate=modified, withdrawn=withdrawn,
                               title=f"Comment {comment_id}")  # fmt: skip
    return entry


def proposal_0001() -> dict[str, Any]:
    """The docket's proposed rule, which the recorded comments are filed on (commentOnDocumentId FAA-2026-2730-0001,
    commentOn 09000064b924b533): its listing entry in the recorded shape, its dates set for these tests."""
    entry = copy.deepcopy(next(d for d in load("documents_search.json")["body"]["data"] if d["id"] == "FAA-2026-2730-0006"))
    entry["id"] = "FAA-2026-2730-0001"
    entry["attributes"].update(
        documentType="Proposed Rule", subtype="Request for Comment", objectId="09000064b924b533", frDocNum="2026-07001",
        postedDate="2026-04-21T04:00:00Z", commentStartDate="2026-04-21T04:00:00Z",
        commentEndDate="2026-06-05T03:59:59Z", lastModifiedDate="2026-04-21T16:02:11Z",
    )  # fmt: skip
    return entry


def fixture_world(api: FakeAPI, *, scanned_0003: bytes | None = None) -> FakeAPI:
    """The recorded docket: its search results, its final rule, the proposal its comments are filed on, its three
    comments with their records, and United's attachment. Boeing's attachment (0003) was not recorded; where a test needs
    it, it serves the given bytes."""
    for d in load("documents_search.json")["body"]["data"]:
        api.add_document(d)
    api.add_document(proposal_0001())
    api.details[f"documents/{DOCKET}-0006"] = load("document.json")["body"]
    api.details[f"dockets/{DOCKET}"] = load("docket.json")["body"]
    details = {load(f"comment_{i}.json")["body"]["data"]["id"]: load(f"comment_{i}.json")["body"] for i in range(3)}
    for entry in load("comments_list.json")["body"]["data"]:
        api.add_comment(DOCKET, entry, details[entry["id"]])
    api.files[PDF_0005] = fixture_pdf()
    if scanned_0003 is not None:
        api.files[PDF_0003] = scanned_0003
    return api


# ---------------------------------------------------------------- the Federal Register


@dataclass
class FakeFederalRegister:
    """The Federal Register's documents endpoint, as it answers a request for several document numbers: each number it
    knows with its stated effective date (None: it states none), the rest as not found."""

    effective: dict[str, str | None] = field(default_factory=dict)  # document number -> effective_on, or None
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.federalregister.gov" and request.headers["user-agent"] == USER_AGENT
        self.requests.append(request)
        numbers = request.url.path.removeprefix("/api/v1/documents/").removesuffix(".json").split(",")
        found = [{"document_number": n, "type": "Rule", "publication_date": None, "effective_on": self.effective[n],
                  "dates": f"This rule is effective {self.effective[n]}." if self.effective[n] else None}
                 for n in numbers if n in self.effective]  # fmt: skip
        missing = [n for n in numbers if n not in self.effective]
        body: dict[str, Any] = {"count": len(found), "results": found}
        if missing:
            body["errors"] = {"not_found": missing}
        return httpx.Response(200, json=body, headers={"content-type": "application/json; charset=utf-8"})


# ---------------------------------------------------------------- PDFs for the OCR path


def _jpeg_size(data: bytes) -> tuple[int, int, int]:
    """Width, height and components of a baseline or progressive JPEG (its SOF segment)."""
    i = 2
    while i < len(data):
        marker = data[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(data[i + 5 : i + 7], "big"), data[i + 9]
        i += 2 + length
    raise ValueError("no frame header in the JPEG")


def image_only_pdf(jpeg: bytes, dpi: int) -> bytes:
    """A one-page PDF whose page is the image and nothing else: no text layer."""
    w, h, components = _jpeg_size(jpeg)
    width, height = w * 72 / dpi, h * 72 / dpi
    content = f"q {width:.2f} 0 0 {height:.2f} 0 0 cm /Im0 Do Q".encode()
    space = "/DeviceGray" if components == 1 else "/DeviceRGB"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /XObject << /Im0 4 0 R"
         " >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {space} /BitsPerComponent 8 "
         f"/Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n").encode() + jpeg + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]  # fmt: skip
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def rasterized(pdf: bytes, page: int = 1, dpi: int = 200) -> bytes:
    """The page rendered by poppler to a grayscale JPEG and wrapped as an image-only PDF: a scan of it."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.pdf"
        source.write_bytes(pdf)
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-jpeg", "-gray", "-singlefile",
                        str(source), str(Path(tmp) / "scan")], check=True, capture_output=True)  # fmt: skip
        return image_only_pdf((Path(tmp) / "scan.jpg").read_bytes(), dpi)

