"""What each measure is, as its own published text states it (core.MeasureText; Owen, November 2: question 2 is asked
over "the agency's summary plus the agency's own cost statements").

A measure with a Federal Register document number is read from the Federal Register: its record (the abstract, the
SUMMARY its published text opens with) and its whole published text. One without is read from its regulations.gov
record as the Mirrulations mirror holds it (its docAbstract; taken only where it is current against the frame's
official listing of the docket's documents, as a comment's copy is) and the document file that record names, from
regulations.gov's file host (no key), read by the PDF reader. The whole text is read by
the cost_statements agent in consecutive parts within its reading limit (PART_CHARS), each part's quotes checked as
text against that part (agents/answers.py CostStatementsDraft); the parts' statements, in the text's order, are the
measure's. Every response is committed as an artifact (spec I2), and each MeasureText is committed once per document and
read thereafter.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import CostStatementsDraft
from requisite.agents.trace import record_run
from requisite.evidence import extract
from requisite.evidence.artifacts import LISTING_BASIS, commit
from requisite.evidence.federal_register import FederalRegister
from requisite.evidence.mirrulations import Mirrulations
from requisite.evidence.regulations_gov import RegulationsGov
from requisite.store import Refused, Store

ROLE = "cost_statements"
# The reader's version: 2 reads each sum's scope and the count of its payers (Owen, November 2). A MeasureText is kept
# per document and reader version (committed state is never overwritten).
READER = f"{ROLE}/2"


def key(document: str) -> str:
    return f"{document}|{READER}"
# The characters of text one reading takes: about 75,000 tokens, well within the reader's context with its prompt and
# answer (ours). A part ends at the last blank line before the limit, so a paragraph is read whole wherever it can be.
PART_CHARS = 300_000


def parts(text: str, limit: int = PART_CHARS) -> list[str]:
    """The text in consecutive parts of at most `limit` characters, each ending at a paragraph break where one falls in
    its second half; together, exactly the text."""
    out, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + limit // 2, end)
            if cut > 0:
                end = cut + 2
        out.append(text[start:end])
        start = end
    return out


def listed(store: Store, measure: core.ItemMeasure) -> Mapping[str, Any]:
    """The measure's document as regulations.gov lists it, read from the records its statement rests on (the frame's
    listing of its docket's documents, or the document's own record): its attributes."""
    for art_id in measure.artifacts:
        artifact = store.get("artifact", art_id, core.Artifact)
        if artifact is None or artifact.source != "regulations.gov":
            continue
        body = json.loads(store.content(art_id))
        data = body.get("data")
        for d in data if isinstance(data, list) else [data] if isinstance(data, dict) else []:
            if d.get("id") == measure.document:
                return d.get("attributes") or {}
    raise Refused(f"no record the measure {measure.document} rests on lists it")


def fr_number(attributes: Mapping[str, Any]) -> str | None:
    """The Federal Register document number regulations.gov states for a document."""
    n = attributes.get("frDocNum")
    return str(n) if n else None


def read_text(store: Store, documents: Mapping[str, Mapping[str, Any]], *, workers: int = 4,
              federal_register: Callable[[], FederalRegister] = FederalRegister,
              mirror: Callable[[], Mirrulations] = Mirrulations,
              regulations_gov: Callable[[], RegulationsGov] = RegulationsGov,
              on_done: Callable[[str], None] | None = None) -> dict[str, core.MeasureText]:  # fmt: skip
    """Each document's MeasureText (document id -> its attributes as regulations.gov's official listing gives them):
    the committed one where it exists, otherwise fetched, read and committed."""
    out: dict[str, core.MeasureText] = {}
    todo = []
    for doc, attributes in sorted(documents.items()):
        have = store.get("measure_text", key(doc), core.MeasureText)
        if have is not None:
            out[doc] = have
        else:
            todo.append((doc, attributes))
    if not todo:
        return out
    fr, mi, rg = federal_register(), mirror(), regulations_gov()
    try:
        sources = {doc: _source(store, doc, attrs, fr, mi, rg) for doc, attrs in todo}
    finally:
        fr.close()
        mi.close()
        rg.close()
    jobs: list[tuple[str, int, int, str]] = []
    for doc, (_, _, _, _, text, _) in sources.items():
        ps = parts(text)
        jobs += [(doc, i, len(ps), part) for i, part in enumerate(ps, 1)]

    def one(job: tuple[str, int, int, str]) -> runner.AgentResult[CostStatementsDraft]:
        doc, i, n, part = job
        return runner.run(ROLE, _input(sources[doc][0], i, n, part), CostStatementsDraft, effort="medium",
                          context={"text": part})  # fmt: skip

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    by_doc: dict[str, list[tuple[tuple[str, int, int, str], runner.AgentResult[CostStatementsDraft]]]] = {}
    for job, res in zip(jobs, results, strict=True):
        by_doc.setdefault(job[0], []).append((job, res))
    for doc, (title, summary, summary_why, arts, _, served) in sources.items():
        read = by_doc.get(doc, [])
        if not read:
            raise Refused(f"{doc}: its published text is empty, so nothing of it was read")
        statements: list[core.CostStatement] = []
        seen: set[str] = set()
        for job, res in read:
            record_run(store, res, _input(title, job[1], job[2], job[3]), [])
            for s in res.output.statements:
                if s.quote in seen:
                    continue
                seen.add(s.quote)
                statements.append(core.CostStatement(**s.model_dump()))
        none_why = None
        if not statements:
            said = "; ".join(f"part {job[1]}: {res.output.none_why}" for job, res in read)
            none_why = f"read whole in {len(read)} part(s), it states no cost to anyone ({said})"
        number = fr_number(documents[doc]) if served else None
        mt = core.MeasureText(
            document=doc, reader=READER, fr_number=number,
            summary=summary if summary else None,
            summary_why=None if summary else summary_why,
            statements=statements, none_why=none_why, artifacts=arts, readings=[res.run for _, res in read],
        )  # fmt: skip
        store.commit("measure_text", key(doc), mt)
        out[doc] = mt
        if on_done is not None:
            on_done(doc)
    return out


def _input(title: str, i: int, n: int, part: str) -> dict[str, Any]:
    return {"measure": title, "part": f"{i} of {n}", "text": part}


def _source(store: Store, doc: str, attributes: Mapping[str, Any], fr: FederalRegister, mi: Mirrulations,
            rg: RegulationsGov) -> tuple[str, str | None, str | None, list[str], str, bool]:  # fmt: skip
    """A document's title, its source's summary or why none is shown, the artifacts read, its whole published text, and
    whether the Federal Register served it."""
    number = fr_number(attributes)
    arts: list[str] = []
    if number is not None:
        record = fr.document(number)
        art = commit(store, record, published_at=None, basis=LISTING_BASIS)
        arts.append(art.id)
    if number is not None and record.status == 200:
        body = record.json()
        if not body.get("raw_text_url"):
            raise Refused(f"{doc}: the Federal Register states no published text for document {number} ({art.id})")
        full = fr.full_text(str(body["raw_text_url"]))
        text_art = commit(store, full, published_at=None, basis=LISTING_BASIS)
        abstract = body.get("abstract")
        text = full.content.decode("utf-8")
        why = None if abstract else "the Federal Register states no abstract for it"
        return str(body.get("title") or doc), str(abstract) if abstract else None, why, [art.id, text_art.id], text, True
    # No number, or one the Federal Register does not serve (its 404, committed): the document's regulations.gov record
    # as the mirror holds it, taken where it is current against the official listing, and the file it names.
    mirrored = mi.document(doc)
    if mirrored is None:
        raise Refused(f"{doc}: the Federal Register does not serve it and the mirror holds no record of it")
    art = commit(store, mirrored, published_at=None, basis=LISTING_BASIS)
    attrs = (mirrored.json().get("data") or {}).get("attributes") or {}
    # The record is a copy: its abstract is shown as the summary only where the copy is current against the official
    # listing (as a comment's copy is taken: requisite/evidence/items.py). Where it is not, the record still says where
    # the document's file is, and the file comes from regulations.gov's own file host, the official copy.
    stale = sorted(k for k, v in attributes.items() if attrs.get(k) != v)
    files = [f for f in attrs.get("fileFormats") or [] if f.get("format") in ("pdf", "htm", "html")]
    if not files:
        raise Refused(f"{doc}: regulations.gov names no file of the document's text ({art.id})")
    content = rg.download(str(files[0]["fileUrl"]))
    file_art = commit(store, content, published_at=None, basis=LISTING_BASIS)
    reader = "pdf" if files[0]["format"] == "pdf" else "html"
    x = extract.extract(reader, file_art.id, content.content)
    if x.unreadable or x.text is None:
        raise Refused(f"{doc}: its document file could not be read whole: {'; '.join(x.unreadable)}")
    abstract = attrs.get("docAbstract") if not stale else None
    why = (None if abstract else
           f"regulations.gov's abstract is known here only from the mirror's copy of its record, which differs from the "
           f"official listing in {', '.join(stale)}" if stale else "regulations.gov states no abstract for it")  # fmt: skip
    return str(attrs.get("title") or doc), str(abstract) if abstract else None, why, [*arts, art.id, file_art.id], x.text, False


def measure_texts(store: Store, documents: Sequence[str]) -> dict[str, core.MeasureText]:
    """The committed MeasureText of each document; refused where one is not yet read (`wide measure-text`)."""
    out = {}
    for doc in documents:
        mt = store.get("measure_text", key(doc), core.MeasureText)
        if mt is None:
            raise Refused(f"the text of measure {doc} is not read yet: run `requisite wide measure-text`")
        out[doc] = mt
    return out
