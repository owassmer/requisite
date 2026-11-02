"""Items: one per comment, read whole (spec §7.2 step 1: "split it into items: comments, letters, filings, posts";
decisions: "An item's text is everything its submitter filed: the comment and the text of each attachment").

A sample's comments are fetched in parallel, each its record with its attachments, every file it filed downloaded and
read (a PDF rendition where the API offers one, else the original by its format), all within one shared pacing state.
Code then commits, in the sample's order: each response as an artifact (its record, published at its postedDate; each
file, published with it), each text as an extraction, and the core Item, whose text is the comment and then each file in
the order filed, each part headed by the artifact it came from. What could not be read (a restricted attachment, a file
the host no longer serves, a format no reader here reads, a corrupt file) is stated in the text where it stands and on
the item's `unreadable`, never dropped.

A comment the frame listed but the API no longer serves (404) stays an item: its artifact is the API's answer, its
docket and posted time are the frame's, and its text says what was found.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from requisite.core import FrameItem, Item
from requisite.core.evidence import artifact_id
from requisite.evidence import extract
from requisite.evidence.artifacts import SOURCE, commit
from requisite.evidence.regulations_gov import Fetched, RegulationsGov, instant
from requisite.store import Store

WORKERS = 8
RECORD_BASIS = "the comment's postedDate (regulations.gov)"
FILE_BASIS = "the postedDate of the comment it is filed with (regulations.gov)"
GONE_BASIS = "none stated: the source's answer that it no longer serves the record"


@dataclass(frozen=True)
class _File:
    label: str
    fetched: Fetched | None
    extraction: extract.Extraction | None
    unreadable: str | None  # why the file itself could not be fetched or has no reader here


@dataclass(frozen=True)
class _Read:
    entry: FrameItem
    record: Fetched
    comment: extract.Extraction | None
    files: tuple[_File, ...]


class _Once:
    """Each key's work done once in a run, however many threads ask for it (two comments filing the same bytes share an
    artifact, so its extraction is one)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[str, Future[Any]] = {}

    def __call__[T](self, key: str, work: Callable[[], T]) -> T:
        with self._lock:
            future = self._futures.get(key)
            owner = future is None
            if future is None:
                future = self._futures[key] = Future()
        if owner:
            try:
                future.set_result(work())
            except BaseException as e:
                future.set_exception(e)
                raise
        return future.result()


def _format(entry: Mapping[str, Any]) -> str:
    fmt = str(entry.get("format") or "").strip().lower()
    return fmt or urlsplit(str(entry.get("fileUrl", ""))).path.rsplit(".", 1)[-1].lower()


def choose(formats: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str | None]:
    """The rendition to read and its reader: the PDF where one is offered, else the first format a reader here reads,
    else the first offered (fetched and kept, its text unread, with why)."""
    for f in formats:
        if _format(f) == "pdf":
            return f, "pdf"
    for f in formats:
        if _format(f) in extract.READERS:
            return f, extract.READERS[_format(f)]
    return formats[0], None


def _restriction(attributes: Mapping[str, Any]) -> str:
    kind, why = attributes.get("restrictReasonType"), attributes.get("restrictReason")
    if kind or why:
        return "restricted by the agency" + "".join(f"; {v}" for v in (kind, why) if v) + "; no file is published"
    return "the record lists no file for it"


def _file(client: RegulationsGov, once: _Once, label: str, attributes: Mapping[str, Any]) -> _File:
    formats = attributes.get("fileFormats") or []
    if not formats:
        return _File(label, None, None, _restriction(attributes))
    chosen, reader = choose(formats)
    url = str(chosen["fileUrl"])
    fetched = client.download(url)
    if fetched.status != 200:
        return _File(label, None, None, f"the file host answered {fetched.status} for {url}")
    if reader is None:
        return _File(label, fetched, None, f".{_format(chosen)}: no reader here, and no PDF rendition is offered")
    aid = artifact_id(fetched.content)
    extraction = once(
        f"{reader}|{aid}",
        lambda: extract.extract(reader, aid, fetched.content, root=client.root, budget=client.budget),
    )
    return _File(label, fetched, extraction, None)


def _fetch(client: RegulationsGov, once: _Once, entry: FrameItem) -> _Read:
    """One comment's record, files and texts (run in a worker thread; nothing here touches the store)."""
    record = client.comment(entry.id)
    if record.status == 404:
        return _Read(entry, record, None, ())
    body = record.json()
    data = body["data"]
    attributes = data.get("attributes") or {}
    rid = artifact_id(record.content)
    comment = once(
        f"comment|{rid}", lambda: extract.extract("comment", rid, record.content, root=client.root, budget=client.budget)
    )
    files: list[_File] = []
    if attributes.get("fileFormats"):
        files.append(_file(client, once, "the comment's own file", attributes))
    included = {str(i["id"]): i for i in body.get("included") or [] if i.get("type") == "attachments"}
    related = (data.get("relationships") or {}).get("attachments") or {}
    listed = [str(r["id"]) for r in related.get("data") or []]
    order = listed + [i for i in included if i not in listed]
    attachments = []
    for position, aid in enumerate(order):
        a = included.get(aid)
        rank = (a or {}).get("attributes", {}).get("docOrder")
        attachments.append((rank if isinstance(rank, int) else position + 1, position, aid, a))
    for rank, _, aid, a in sorted(attachments, key=lambda x: (x[0], x[1])):
        if a is None:
            files.append(_File(f"attachment {rank} ({aid})", None, None, "the record names it but does not include it"))
            continue
        title = (a.get("attributes") or {}).get("title")
        label = f'attachment {rank} "{title}"' if title else f"attachment {rank}"
        files.append(_file(client, once, label, a.get("attributes") or {}))
    return _Read(entry, record, comment, tuple(files))


def submitter(attributes: Mapping[str, Any]) -> str | None:
    """The name the submitter gave, as published: the organization they represent, else the agency, else their name;
    None where the source publishes none."""
    for key in ("organization", "govAgency"):
        value = str(attributes.get(key) or "").strip()
        if value:
            return value
    name = " ".join(str(attributes.get(k) or "").strip() for k in ("firstName", "lastName")).strip()
    return name or None


def _item(store: Store, read: _Read) -> Item:
    """Commit one comment's artifacts and extractions, and its item."""
    entry = read.entry
    if read.comment is None:  # the API no longer serves the record
        gone = commit(store, read.record, published_at=None, basis=GONE_BASIS)
        note = (f"the comment's record: regulations.gov answered {read.record.status} for it at "
                f"{gone.observed_at.isoformat()}; it was listed in the frame and is no longer served")  # fmt: skip
        item = Item(id=entry.id, source=SOURCE, container=entry.docket, posted_at=entry.posted_at, submitter=None,
                    artifacts=[gone.id], text=f"[comment {entry.id} · {gone.id}]\n({note})", extractions=[],
                    unreadable=[note])  # fmt: skip
        store.commit("item", f"{item.id}|{item.content_hash()}", item)
        return item
    data = read.record.json()["data"]
    attributes = data.get("attributes") or {}
    posted = instant(str(attributes["postedDate"]))
    record = commit(store, read.record, published_at=posted, basis=RECORD_BASIS)
    artifacts, extractions, parts, unreadable = [record.id], [], [], []
    if read.comment.id is None or read.comment.text is None:
        raise ValueError(f"comment {entry.id}: its record has no readable comment field")
    store.commit_extraction(record.id, read.comment.id, read.comment.text)
    extractions.append(read.comment.id)
    parts.append(f"[comment {data['id']} · {record.id}]\n{read.comment.text}".rstrip("\n"))
    for f in read.files:
        if f.fetched is None:
            parts.append(f"[{f.label}]\n(its text could not be read: {f.unreadable})")
            unreadable.append(f"{f.label}: {f.unreadable}")
            continue
        art = commit(store, f.fetched, published_at=posted, basis=FILE_BASIS)
        artifacts.append(art.id)
        heading = f"[{f.label} · {urlsplit(f.fetched.locator).path.rsplit('/', 1)[-1]} · {art.id}]"
        x = f.extraction
        if x is None or x.id is None or x.text is None:
            why = f.unreadable or "; ".join(x.unreadable if x is not None else ())
            parts.append(f"{heading}\n(its text could not be read: {why})")
            unreadable.append(f"{f.label}: {why}")
            continue
        store.commit_extraction(art.id, x.id, x.text)
        extractions.append(x.id)
        parts.append(f"{heading}\n{x.text}".rstrip("\n"))
        unreadable.extend(f"{f.label}: {u}" for u in x.unreadable)
    item = Item(
        id=str(data["id"]),
        source=SOURCE,
        container=str(attributes["docketId"]),
        posted_at=posted,
        submitter=submitter(attributes),
        artifacts=artifacts,
        text="\n\n".join(parts),
        extractions=extractions,
        unreadable=unreadable,
    )
    store.commit("item", f"{item.id}|{item.content_hash()}", item)
    return item


def read_items(client: RegulationsGov, store: Store, entries: Sequence[FrameItem], *,
               workers: int = WORKERS) -> list[Item]:  # fmt: skip
    """The full items of these frame entries, in their order: fetched in parallel within the rate limit, committed here."""
    once = _Once()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reads = list(pool.map(lambda e: _fetch(client, once, e), entries))
    return [_item(store, r) for r in reads]
