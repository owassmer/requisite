"""A wide run's frame, from its criteria and nothing else, frozen (spec §7.2 step 1, §7.3; decisions, "Discovery finds
what it analyzes": the criteria come from the approved objective, and which dockets the frame reaches is its finding).

The criteria run as regulations.gov's own searches: for each search term, each document type, and each agency where
they name any, the documents posted in the window that match it; the results are united and grouped by docket. Each
docket reached is kept when its comment status is as asked and it has at least min_comments public comments:
- open: a document's comment period is running at as_of (it has started, where a start is given, and ends after as_of);
- closed_pending_final: its proposal is its latest Proposed Rule posted by as_of (a docket with none is dropped with that
  reason), every comment period has ended by as_of, and no Rule was posted after the proposal;
- either: one or the other.
Every test reads only what was posted by as_of, decided for each document and comment by core.observable_at over the
listing that served it (spec I3): the API serves today's record, so a document, a comment or a final rule posted after
as_of is excluded from each test, never read as known then. A document the API lists as withdrawn is not counted as
posted (the API's documentation: "If set to true, the document is withdrawn otherwise it's a posted document"). The
frame's items are every comment of the kept dockets posted by as_of, from the comment listing (no per-comment request),
each with its id, docket and posted time; a comment the API lists as withdrawn stays, marked, since it was public then.

Time. The frame's retrieval starts and ends at the first and last responses it rests on, as their cassettes recorded
them, so a replay reproduces the same frame. A frame asked for an as_of before its retrieval ended is a retrospective
reconstruction at that as_of; asked for any later instant, it is prospective, and its as_of is the instant its
retrieval ended, since nothing after it was observed (spec I3). The mode follows from the times by core.Frame's rule.

The API filters posting dates by day (its OpenAPI specification), so a request is bounded by the day after as_of, the
widest any time zone reads it, and as_of itself is settled entry by entry by observable_at. Every response the frame
rests on (searches, listings) is committed as an artifact, and the frame names them, so its counts recompute from them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from requisite import core
from requisite.core import (
    Artifact,
    Frame,
    FrameCriteria,
    FrameDocket,
    FrameItem,
    FrameSearch,
    GateResult,
    observable_at,
)
from requisite.evidence.artifacts import LISTING_BASIS, commit, listed
from requisite.evidence.regulations_gov import Entry, Listing, RegulationsGov, identity, instant
from requisite.store import Store

WORKERS = 8
POSTED = "the entry's postedDate in the listing"


@dataclass(frozen=True)
class _Search:
    term: str
    listing: Listing


def _day_bound(criteria: FrameCriteria) -> str:
    """The last posting day a request asks for: the window's end, and never past the day after as_of."""
    bound = criteria.as_of.astimezone(UTC).date() + timedelta(days=1)
    if criteria.posted_to is not None:
        bound = min(bound, criteria.posted_to)
    return bound.isoformat()


def _searches(client: RegulationsGov, criteria: FrameCriteria) -> list[_Search]:
    out = []
    for term in criteria.search_terms:
        for kind in criteria.document_types:
            agencies: list[str | None] = [*criteria.agencies] or [None]
            for agency in agencies:
                filters = {
                    "filter[searchTerm]": term,
                    "filter[documentType]": kind,
                    "filter[postedDate][ge]": criteria.posted_from.isoformat(),
                    "filter[postedDate][le]": _day_bound(criteria),
                }
                if agency is not None:
                    filters["filter[agencyId]"] = agency
                out.append(_Search(term, client.documents(filters)))
    return out


def _docket_filters(docket: str, bound: str) -> dict[str, str]:
    return {"filter[docketId]": docket, "filter[postedDate][le]": bound}


class _Evidence:
    """The committed artifacts of a run's listings, and each entry as evidence for observable_at."""

    def __init__(self, store: Store):
        self.store = store
        self._pages: dict[int, list[Artifact]] = {}

    def pages(self, listing: Listing) -> list[Artifact]:
        if id(listing) not in self._pages:
            self._pages[id(listing)] = [
                commit(self.store, p, published_at=None, basis=LISTING_BASIS) for p in listing.pages
            ]
        return self._pages[id(listing)]

    def posted(self, listing: Listing, as_of: datetime, evaluated: datetime, mode: str) -> list[Entry]:
        """The entries of a listing posted by as_of, decided by core.observable_at (spec I3)."""
        pages = self.pages(listing)
        return [e for e in listing.entries
                if observable_at(listed(pages[e.page], instant(str(e.attributes["postedDate"])), POSTED), as_of, evaluated, mode)]  # fmt: skip


def _when(value: Any) -> datetime | None:
    return instant(str(value)) if value else None


def _runs(d: Entry, as_of: datetime) -> bool:
    """Whether a document's comment period runs at as_of: it has started, where a start is given, and ends after."""
    end, start = _when(d.attributes.get("commentEndDate")), _when(d.attributes.get("commentStartDate"))
    return end is not None and end > as_of and (start is None or start <= as_of)


def _open(documents: Sequence[Entry], as_of: datetime) -> tuple[bool, str]:
    for d in documents:
        if _runs(d, as_of):
            end = _when(d.attributes.get("commentEndDate"))
            assert end is not None  # _runs
            return True, f"open: {d.id}'s comment period ends {end.isoformat()}, after as_of"
    return False, "not open: no document posted by as_of has a comment period running at as_of"


def _pending(documents: Sequence[Entry], as_of: datetime) -> tuple[bool, str]:
    proposals = [d for d in documents if d.attributes.get("documentType") == "Proposed Rule"]
    if not proposals:
        return False, "not closed pending a final rule: no proposed rule posted by as_of"
    proposal = max(proposals, key=lambda d: (instant(str(d.attributes["postedDate"])), d.id))
    proposed = instant(str(proposal.attributes["postedDate"]))
    for d in documents:
        end = _when(d.attributes.get("commentEndDate"))
        if end is not None and end > as_of:
            return False, f"not closed: {d.id}'s comment period ends {end.isoformat()}, after as_of"
    for d in documents:
        if d.attributes.get("documentType") == "Rule" and instant(str(d.attributes["postedDate"])) > proposed:
            return False, (f"not pending a final rule: {d.id} (Rule) was posted {d.attributes['postedDate']}, after "
                           f"the proposal {proposal.id}")  # fmt: skip
    return True, (f"closed pending a final rule: every comment period ended by as_of and no Rule follows the proposal "
                  f"{proposal.id}, posted {proposal.attributes['postedDate']}")  # fmt: skip


def _status(asked: str, documents: Sequence[Entry], as_of: datetime) -> tuple[bool, str]:
    if asked == "open":
        return _open(documents, as_of)
    if asked == "closed_pending_final":
        return _pending(documents, as_of)
    is_open, why_open = _open(documents, as_of)
    if is_open:
        return True, why_open
    pending, why_pending = _pending(documents, as_of)
    return pending, why_pending if pending else f"{why_open}; {why_pending}"


def take_frame(criteria: FrameCriteria, client: RegulationsGov, store: Store, *, workers: int = WORKERS) -> Frame:
    """Run the criteria, freeze what they return as a core.Frame, and commit it with every response it rests on.

    Two stages: the searches and each candidate docket's documents, then the comments. Where as_of is already past when
    the first stage ends, the membership test's instant is fixed, and only the dockets the searches reached by as_of and
    whose comment status passes are listed for comments (core.FrameDocket: a dropped docket's count decides nothing).
    Otherwise as_of is the end of the retrieval, not yet known, and every candidate's comments are listed."""
    if criteria.source != "regulations.gov":
        raise ValueError(f"no adapter reads {criteria.source}")
    searches = _searches(client, criteria)
    candidates = sorted({str(e.attributes["docketId"]) for s in searches for e in s.listing.entries
                         if e.attributes.get("docketId")})  # fmt: skip
    bound = _day_bound(criteria)
    evidence = _Evidence(store)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        documents = dict(zip(candidates, pool.map(lambda d: client.documents(_docket_filters(d, bound)), candidates),
                             strict=True))  # fmt: skip
        first = [p for s in searches for p in s.listing.pages] + [p for x in documents.values() for p in x.pages]
        if not first:
            raise ValueError("the criteria made no request")
        first_ended = max(p.retrieved_at for p in first)
        if criteria.as_of < first_ended:
            decided = _decide(criteria, searches, documents, evidence, criteria.as_of, first_ended)
            listed = [d for d, (status, _, _) in decided.items() if status]
        else:
            listed = candidates
        comments = dict(
            zip(listed, pool.map(lambda d: client.comments(_docket_filters(d, bound)), listed), strict=True)
        )
    responses = first + [p for x in comments.values() for p in x.pages]
    started = min(p.retrieved_at for p in responses)
    ended = max(p.retrieved_at for p in responses)
    as_of = min(criteria.as_of, ended)
    mode: Literal["prospective", "retrospective_reconstruction"] = (
        "prospective" if ended <= as_of else "retrospective_reconstruction"
    )
    for listing in (*(s.listing for s in searches), *documents.values(), *comments.values()):
        evidence.pages(listing)  # every response the retrieval made is kept, so its times recompute from artifacts

    frame_searches = [FrameSearch(term=s.term, request=s.listing.request, documents=len(s.listing.entries),
                                  artifacts=[a.id for a in evidence.pages(s.listing)]) for s in searches]  # fmt: skip
    frame_dockets: list[FrameDocket] = []
    items: list[FrameItem] = []
    for d, (status, why, withdrawn) in _decide(criteria, searches, documents, evidence, as_of, ended).items():
        set_aside = f"; {withdrawn} withdrawn document(s) not counted as posted" if withdrawn else ""
        doc_artifacts = [a.id for a in evidence.pages(documents[d])]
        if d not in comments:  # dropped by its comment status before its comments were listed (core.FrameDocket)
            assert not status, f"{d} passed its status at as_of but its comments were not listed"
            frame_dockets.append(FrameDocket(docket=d, kept=False, artifacts=doc_artifacts, comments=None,
                                             reason=f"{why}; its comments were not listed, since its status drops it"
                                                    f"{set_aside}"))  # fmt: skip
            continue
        posted = evidence.posted(comments[d], as_of, ended, mode)
        enough = len(posted) >= criteria.min_comments
        count = (f"{len(posted)} public comments posted by as_of"
                 f"{'' if enough else f', fewer than the {criteria.min_comments} asked'}")  # fmt: skip
        frame_dockets.append(FrameDocket(docket=d, kept=status and enough, reason=f"{why}; {count}{set_aside}",
                                         comments=len(posted),
                                         artifacts=doc_artifacts + [a.id for a in evidence.pages(comments[d])]))  # fmt: skip
        if status and enough:
            items += [FrameItem(id=c.id, docket=d, posted_at=instant(str(c.attributes["postedDate"])),
                                withdrawn=bool(c.attributes.get("withdrawn"))) for c in posted]  # fmt: skip
    items.sort(key=lambda i: (i.docket, i.posted_at, i.id))
    stamp = ended.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    frame = Frame(
        id=f"frame-{criteria.id}-v{criteria.version}-{stamp}",
        criteria=core.Ref(id=criteria.id, version=criteria.version),
        as_of=as_of,
        retrieval_started=started,
        retrieval_ended=ended,
        mode=mode,
        searches=frame_searches,
        dockets=frame_dockets,
        items=items,
    )
    store.commit("frame", frame.id, frame)
    return frame


def _decide(
    criteria: FrameCriteria,
    searches: Sequence[_Search],
    documents: Mapping[str, Listing],
    evidence: _Evidence,
    as_of: datetime,
    evaluated: datetime,
) -> dict[str, tuple[bool, str, int]]:
    """Each docket the searches reached by as_of, with whether its comment status passes, why, and how many of its
    documents posted by as_of are withdrawn (and so not counted as posted). A docket reached only by documents posted
    after as_of is not reached as of as_of."""
    mode = "prospective" if evaluated <= as_of else "retrospective_reconstruction"
    reached = {str(e.attributes["docketId"]) for s in searches for e in evidence.posted(s.listing, as_of, evaluated, mode)
               if e.attributes.get("docketId")}  # fmt: skip
    out = {}
    for d in sorted(reached & set(documents)):
        posted = evidence.posted(documents[d], as_of, evaluated, mode)
        withdrawn = [e for e in posted if e.attributes.get("withdrawn")]
        status, why = _status(criteria.comment_status, [e for e in posted if not e.attributes.get("withdrawn")], as_of)
        out[d] = (status, why, len(withdrawn))
    return out


def docket_documents(store: Store, frame: Frame, docket: FrameDocket) -> list[Entry]:
    """A frame docket's documents posted by the frame's as_of, withdrawn ones set aside, rebuilt from the listing
    artifacts the frame names for it, each as the last page that served it gave it (decided as take_frame decides)."""
    pages: list[Artifact] = []
    seen: dict[tuple[str, str, str], Entry] = {}
    for art_id in docket.artifacts:
        artifact = store.get("artifact", art_id, Artifact)
        if artifact is None:
            raise ValueError(f"frame {frame.id} names artifact {art_id}, which is not committed")
        for e in json.loads(store.content(art_id)).get("data") or []:
            if e.get("type") == "documents":
                seen[identity(e)] = Entry(e, len(pages))
        pages.append(artifact)
    return [e for e in seen.values() if not e.attributes.get("withdrawn")
            and observable_at(listed(pages[e.page], instant(str(e.attributes["postedDate"])), POSTED), frame.as_of,
                              frame.retrieval_ended, frame.mode)]  # fmt: skip


def measure_document(documents: Sequence[Entry], as_of: datetime) -> Entry | None:
    """The pending measure a kept docket's comments answer (decisions, "F4's sample design"): its latest Proposed Rule
    posted by as_of, or where it has none, the latest-posted document whose comment period runs at as_of. `documents`
    are the docket's documents posted by as_of, withdrawn ones set aside (docket_documents)."""
    proposals = [d for d in documents if d.attributes.get("documentType") == "Proposed Rule"]
    pool = proposals or [d for d in documents if _runs(d, as_of)]
    return max(pool, key=lambda d: (instant(str(d.attributes["postedDate"])), d.id)) if pool else None


def frames(store: Store) -> list[Frame]:
    """The case's frozen frames, oldest first."""
    return store.all("frame", Frame)


def entries(store: Store, ids: Sequence[str]) -> list[FrameItem]:
    """The frame entries of these comment ids, from the case's frames (the latest frame that lists each), in the order
    given: a sample is drawn from a frozen frame."""
    known: dict[str, FrameItem] = {}
    for f in frames(store):
        known.update({i.id: i for i in f.items})
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(f"{len(missing)} id(s) are in no frame of this case (a sample is drawn from a frozen frame): "
                         f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")  # fmt: skip
    return [known[i] for i in ids]


def frame_gate(client: Callable[[], RegulationsGov]) -> Callable[[Store, core.Proposal], GateResult]:
    """The tests gate of a frame proposal (core.REQUIRED_GATES: "frame (its criteria run and freeze a frame)"): the
    criteria run and freeze a frame; it passes when the frame holds an item to sample."""

    def check(store: Store, proposal: core.Proposal) -> GateResult:
        criteria = FrameCriteria.model_validate_json(core.canonical_json(dict(proposal.payload)))
        regulations_gov = client()
        try:
            frame = take_frame(criteria, regulations_gov, store)
        finally:
            regulations_gov.close()
        kept = sum(d.kept for d in frame.dockets)
        detail = (f"frame {frame.id} ({frame.mode}, as of {frame.as_of.isoformat()}): {len(frame.dockets)} dockets "
                  f"reached, {kept} kept, {len(frame.items)} items")  # fmt: skip
        return GateResult(
            gate="tests", passed=bool(frame.items), detail=detail if frame.items else f"{detail}: none to sample"
        )

    return check


def summary(frame: Frame) -> Mapping[str, Any]:
    """The frame as the CLI shows it."""
    return {
        "frame": frame.id,
        "as_of": frame.as_of.isoformat(),
        "mode": frame.mode,
        "retrieval": [frame.retrieval_started.isoformat(), frame.retrieval_ended.isoformat()],
        "searches": [{"term": s.term, "request": s.request, "documents": s.documents} for s in frame.searches],
        "dockets": [
            {"docket": d.docket, "kept": d.kept, "comments": d.comments, "reason": d.reason} for d in frame.dockets
        ],
        "items": len(frame.items),
        "withdrawn": sum(i.withdrawn for i in frame.items),
    }
