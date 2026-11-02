"""Items: one per comment, read whole (spec §7.2 step 1: "split it into items: comments, letters, filings, posts";
decisions: "An item's text is everything its submitter filed: the comment and the text of each attachment").

A sample's comments are fetched in parallel, each its record with its attachments, every file it filed downloaded and
read (a PDF rendition where the API offers one, else the original by its format), all within one shared pacing state.
Code then commits, in the sample's order: each response as an artifact (its record, published at its postedDate; each
file, published with it), each text as an extraction, and the core Item, whose text is the comment and then each file in
the order filed, each part headed by the artifact it came from. What could not be read (a restricted attachment, a file
the host no longer serves, a format no reader here reads, a corrupt file, an empty text) is stated in the text where it
stands and on its part's reading status (core.ItemPart), never dropped. The document the comment addresses is kept as
the source states it (commentOnDocumentId).

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

from requisite.core import FrameItem, Item, ItemPart
from requisite.core.evidence import artifact_id
from requisite.evidence import extract
from requisite.evidence.artifacts import commit
from requisite.evidence.mirrulations import Mirrulations
from requisite.evidence.regulations_gov import SOURCE, Fetched, RegulationsGov, instant
from requisite.store import Store

WORKERS = 8
RECORD_BASIS = "the comment's postedDate (regulations.gov)"
MIRROR_BASIS = "the comment's postedDate, as its record states it (mirrored from regulations.gov)"
# What a copy of a comment's record must carry, equal to the frame's official listing entry, to be taken as the current
# record (with every other field the two share): each record field and the listing field it answers to. A record's
# modifyDate is its listing's lastModifiedDate (checked November 2 on the fixture's three comments and six of November 1,
# 2026); its docket is the frame entry's.
CURRENT = {"postedDate": "postedDate", "withdrawn": "withdrawn", "modifyDate": "lastModifiedDate"}
FILE_BASIS = "the postedDate of the comment it is filed with (regulations.gov)"
GONE_BASIS = "none stated: the source's answer that it no longer serves the record"


@dataclass(frozen=True)
class _File:
    label: str
    fetched: Fetched | None
    extraction: extract.Extraction | None
    unreadable: str | None  # why the file itself could not be fetched or has no reader here
    by: str | None = None  # "source" or "reader" (core.ItemPart.by), where unreadable


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
        return _File(label, None, None, _restriction(attributes), "source")
    chosen, reader = choose(formats)
    url = str(chosen["fileUrl"])
    fetched = client.download(url)
    if fetched.status != 200:
        return _File(label, None, None, f"the file host answered {fetched.status} for {url}", "source")
    if reader is None:
        return _File(label, fetched, None, f".{_format(chosen)}: no reader here, and no PDF rendition is offered", "reader")
    aid = artifact_id(fetched.content)
    extraction = once(
        f"{reader}|{aid}",
        lambda: extract.extract(reader, aid, fetched.content, root=client.root, budget=client.budget),
    )
    return _File(label, fetched, extraction, None)


def current(record: Fetched, entry: FrameItem, listed: Mapping[str, Any]) -> bool:
    """Whether a copy of a comment's record is the current one: its id and docket are the frame entry's, it carries
    every field of CURRENT equal to the official listing's, and every field it shares with the listing entry is equal."""
    try:
        data = record.json()["data"]
    except (ValueError, KeyError, TypeError):
        return False
    attributes = data.get("attributes") or {}
    if data.get("id") != entry.id or attributes.get("docketId") != entry.docket:
        return False
    if any(r not in attributes or f not in listed or attributes[r] != listed[f] for r, f in CURRENT.items()):
        return False
    return all(attributes[k] == v for k, v in listed.items() if k in attributes)


def _record(client: RegulationsGov, mirror: Mirrulations | None, listed: Mapping[str, Mapping[str, Any]],
            entry: FrameItem) -> Fetched:  # fmt: skip
    """A comment's record: the mirror's copy where it is current against the official listing (`current`), else the
    API's."""
    if mirror is not None and entry.id in listed:
        copy = mirror.comment(entry.id)
        if copy is not None and current(copy, entry, listed[entry.id]):
            return copy
    return client.comment(entry.id)


def _fetch(client: RegulationsGov, once: _Once, entry: FrameItem, mirror: Mirrulations | None = None,
           official: Mapping[str, Mapping[str, Any]] | None = None) -> _Read:  # fmt: skip
    """One comment's record, files and texts (run in a worker thread; nothing here touches the store)."""
    record = _record(client, mirror, official or {}, entry)
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
            files.append(_File(f"attachment {rank} ({aid})", None, None, "the record names it but does not include it",
                               "source"))  # fmt: skip
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
    """Commit one comment's artifacts and extractions, and its item: each part with its reading status (core.ItemPart),
    a part whose text is empty recorded as unreadable for that reason, its extraction still committed as what was read."""
    entry = read.entry
    if read.comment is None:  # the API no longer serves the record
        gone = commit(store, read.record, published_at=None, basis=GONE_BASIS)
        note = (f"regulations.gov answered {read.record.status} for it at {gone.observed_at.isoformat()}; it was listed "
                "in the frame and is no longer served")  # fmt: skip
        part = ItemPart(label="the comment's record", locator=read.record.locator, artifact=gone.id, extraction=None,
                        unreadable=note, by="source")  # fmt: skip
        item = Item(id=entry.id, source=SOURCE, container=entry.docket, addresses=None, posted_at=entry.posted_at,
                    submitter=None, parts=[part], text=f"[comment {entry.id} · {gone.id}]\n(the comment's record: {note})")  # fmt: skip
        store.commit("item", f"{item.id}|{item.content_hash()}", item)
        return item
    data = read.record.json()["data"]
    attributes = data.get("attributes") or {}
    posted = instant(str(attributes["postedDate"]))
    basis = MIRROR_BASIS if read.record.source != SOURCE else RECORD_BASIS
    record = commit(store, read.record, published_at=posted, basis=basis)
    if read.comment.id is None or read.comment.text is None:
        raise ValueError(f"comment {entry.id}: the comment reader returned no text for its record")
    store.commit_extraction(record.id, read.comment.id, read.comment.text)
    parts: list[ItemPart] = []
    texts: list[str] = []
    head = f"[comment {data['id']} · {record.id}]"
    if read.comment.text.strip():
        parts.append(ItemPart(label="the comment", locator=read.record.locator, artifact=record.id,
                              extraction=read.comment.id, unreadable=None, by=None))  # fmt: skip
        texts.append(f"{head}\n{read.comment.text}".rstrip("\n"))
    else:
        parts.append(ItemPart(label="the comment", locator=read.record.locator, artifact=record.id, extraction=None,
                              unreadable="its comment field is empty", by="source"))  # fmt: skip
        texts.append(f"{head}\n(its comment field is empty)")
    for f in read.files:
        if f.fetched is None:
            parts.append(ItemPart(label=f.label, locator=None, artifact=None, extraction=None, unreadable=f.unreadable,
                                  by="source"))  # fmt: skip
            texts.append(f"[{f.label}]\n(its text could not be read: {f.unreadable})")
            continue
        art = commit(store, f.fetched, published_at=posted, basis=FILE_BASIS)
        heading = f"[{f.label} · {urlsplit(f.fetched.locator).path.rsplit('/', 1)[-1]} · {art.id}]"
        x = f.extraction
        if x is None or x.id is None or x.text is None:
            why = f.unreadable or "; ".join(x.unreadable if x is not None else ()) or "no reader returned text"
            parts.append(ItemPart(label=f.label, locator=f.fetched.locator, artifact=art.id, extraction=None,
                                  unreadable=why, by="reader" if f.by == "reader" else "source"))  # fmt: skip
            texts.append(f"{heading}\n(its text could not be read: {why})")
            continue
        store.commit_extraction(art.id, x.id, x.text)
        if not x.text.strip():
            why = "; ".join(("no text was found in it", *x.unreadable))
            parts.append(ItemPart(label=f.label, locator=f.fetched.locator, artifact=art.id, extraction=None,
                                  unreadable=why, by="source"))  # fmt: skip
            texts.append(f"{heading}\n(its text could not be read: {why})")
            continue
        partial = "; ".join(x.unreadable) or None
        parts.append(ItemPart(label=f.label, locator=f.fetched.locator, artifact=art.id, extraction=x.id,
                              unreadable=partial, by="source" if partial else None))  # fmt: skip
        texts.append(f"{heading}\n{x.text}".rstrip("\n"))
    addressed = attributes.get("commentOnDocumentId")
    item = Item(
        id=str(data["id"]),
        source=SOURCE,
        container=str(attributes["docketId"]),
        addresses=str(addressed) if addressed else None,
        posted_at=posted,
        submitter=submitter(attributes),
        parts=parts,
        text="\n\n".join(texts),
    )
    store.commit("item", f"{item.id}|{item.content_hash()}", item)
    return item


def read_items(client: RegulationsGov, store: Store, entries: Sequence[FrameItem], *, workers: int = WORKERS,
               mirror: Mirrulations | None = None, official: Mapping[str, Mapping[str, Any]] | None = None
               ) -> list[Item]:  # fmt: skip
    """The full items of these frame entries, in their order: fetched in parallel within the rate limit, committed here.
    With a mirror, each record is the mirror's copy where it is current against `official`, the frame's official listing
    entries (frame.listed_comments), and the API's otherwise."""
    once = _Once()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reads = list(pool.map(lambda e: _fetch(client, once, e, mirror, official), entries))
    return [_item(store, r) for r in reads]


def committed(store: Store, ids: Sequence[str]) -> dict[str, Item]:
    """The items read for these frame entries, by id: each as first read, or as read again from the same retrieved bytes
    (the same artifacts) by a newer reader (reextracted). An item fetched again is a new record that never replaces the
    first, so a sample's filing is the one frozen when the sample was read; a better reading of those same bytes is that
    filing still (November 2: a PDF text layer with no Unicode map had been read as glyph codes)."""
    wanted, out = set(ids), dict[str, Item]()
    for item in store.all("item", Item):  # oldest first
        if item.id in wanted:
            first = out.get(item.id)
            if first is None or tuple(item.artifacts) == tuple(first.artifacts):
                out[item.id] = item
    return out


def _reader_of(extraction: str) -> tuple[str, int]:
    """The reader and its version an extraction id names ("pdf/1 pypdf 6.21.4" -> ("pdf", 1))."""
    head = extraction.split(" ", 1)[0]
    reader, _, version = head.partition("/")
    return reader, int(version)


@dataclass(frozen=True)
class Stale:
    """A part of an item read by an older version of its reader than the one now in use (extract.VERSIONS)."""

    part: int
    reader: str
    artifact: str
    content: bytes
    old: str


def stale(store: Store, item: Item) -> list[Stale]:
    """The parts of an item that a reader now in use would read again (the store is read here, in one thread)."""
    out = []
    for n, p in enumerate(item.parts):
        if p.extraction is None or p.artifact is None:
            continue
        reader, version = _reader_of(p.extraction)
        if reader in extract.VERSIONS and version < extract.VERSIONS[reader]:
            old = store.extraction(p.artifact, p.extraction)
            if old is not None:
                out.append(Stale(n, reader, p.artifact, store.content(p.artifact), old))
    return out


def reextracted(store: Store, item: Item, read: Mapping[int, extract.Extraction]) -> Item | None:
    """The item read again from the bytes it was first read from, where a reader now in use reads one of its files
    differently than the older version that read it (`read`: each stale part's new extraction, by part); None where
    every part reads as before. A part that reads the same keeps its extraction, so its record stays true of what read
    it. Committed here, in the store's thread."""
    parts, text, changed = list(item.parts), item.text, False
    for n, x in read.items():
        p = parts[n]
        assert p.artifact is not None and p.extraction is not None
        old = store.extraction(p.artifact, p.extraction)
        if old is None or x.id is None or x.text is None or x.text == old:
            continue  # nothing better read: the part stands as first read
        store.commit_extraction(p.artifact, x.id, x.text)
        partial = "; ".join(x.unreadable) or None
        parts[n] = p.model_copy(update={"extraction": x.id, "unreadable": partial, "by": "source" if partial else None})
        text = text.replace(old.rstrip("\n"), x.text.rstrip("\n"), 1)
        changed = True
    if not changed:
        return None
    again = Item.model_validate_json(item.model_copy(update={"parts": parts, "text": text}).model_dump_json())
    store.commit("item", f"{again.id}|{again.content_hash()}", again)
    return again
