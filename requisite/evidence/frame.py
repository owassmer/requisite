"""A wide run's frame, from its criteria and nothing else, frozen (spec §7.2 step 1, §7.3; decisions, "Discovery finds
what it analyzes": the criteria come from the approved objective, and which dockets the frame reaches is its finding).

The criteria run as regulations.gov's own searches: for each search term, each document type, and each agency where
they name any, the documents posted in the window that match it; the results are united and grouped by docket. Each
docket reached is kept when its comment status is as asked and it has at least min_comments public comments:
- open: a document's comment period is running at as_of (it has started, where a start is given, and ends after as_of);
- closed_pending_final: its proposal is its latest Proposed Rule posted by as_of (a docket with none is dropped with that
  reason), every comment period has ended by as_of, and the measure is still pending as the objective's events define
  pending: no Rule was posted after the proposal (proposed, not final), or the first Rule posted after it takes effect
  after as_of (final, not yet in effect), by the effective date the Federal Register states for it
  (requisite/evidence/federal_register.py; regulations.gov states none). A rule whose effective date the Federal
  Register does not state leaves the measure's status unknown, never "in effect": the docket is kept, saying so (spec
  I6), and each of its items is asked about its measure with that status stated (core.CommentStatus);
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
rests on (searches, listings, the Federal Register's statements of effective dates) is committed as an artifact, and the
frame names them, so its counts and statuses recompute from them.

Reach and meaning. A docket is the unit the frame reaches: its status is decided by its latest proposal and the first
rule after it, so in a docket that holds several rulemakings another rulemaking's status can decide whether the docket is
reached. What each item is asked about is never the docket's: it is the document the item addresses, with that
document's own status (measure_of), so an item is judged against its own measure (external evaluation, November 2).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

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
from requisite.evidence.federal_register import SOURCE as FEDERAL_REGISTER
from requisite.evidence.federal_register import FederalRegister, Published, asked_in, stated
from requisite.evidence.regulations_gov import Entry, Listing, RegulationsGov, identity, instant
from requisite.store import Store

WORKERS = 8
POSTED = "the entry's postedDate in the listing"
EASTERN = ZoneInfo("America/New_York")  # the Federal Register's dates are days in Washington


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


def _posted(d: Entry) -> datetime:
    return instant(str(d.attributes["postedDate"]))


def latest_proposal(documents: Sequence[Entry]) -> Entry | None:
    proposals = [d for d in documents if d.attributes.get("documentType") == "Proposed Rule"]
    return max(proposals, key=lambda d: (_posted(d), d.id)) if proposals else None


def final_rule(documents: Sequence[Entry], proposal: Entry) -> Entry | None:
    """The first Rule posted after a proposal in its docket's documents: the one that makes it final."""
    rules = [d for d in documents if d.attributes.get("documentType") == "Rule" and _posted(d) > _posted(proposal)]
    return min(rules, key=lambda d: (_posted(d), d.id)) if rules else None


def rule_status(rule: Entry, as_of: datetime, published: Mapping[str, Published]) -> tuple[bool | None, str]:
    """Whether a final rule is in effect by as_of (False: final, not yet in effect; None: not known), and why, from the
    effective date the Federal Register states for it. A day the rule takes effect is compared with as_of's day in
    Washington, where the Federal Register dates it."""
    number = rule.attributes.get("frDocNum")
    named = f"{rule.id} (Rule, posted {str(rule.attributes['postedDate'])[:10]})"
    if not number:
        return None, f"{named} names no Federal Register document number, so its effective date is not known"
    p = published.get(str(number))
    if p is None:
        raise ValueError(f"the Federal Register was not asked about {number}, the rule {rule.id}")
    if not p.found:
        return None, f"{named}: the Federal Register does not find {number}, so its effective date is not known"
    if p.effective_on is None:
        return None, f"{named}: the Federal Register ({number}) states no effective date"
    day = as_of.astimezone(EASTERN).date()
    if p.effective_on > day:
        return False, f"{named} takes effect {p.effective_on.isoformat()} (Federal Register {number}), after as_of"
    return True, f"{named} took effect {p.effective_on.isoformat()} (Federal Register {number}), by as_of"


def _pending(documents: Sequence[Entry], as_of: datetime, published: Mapping[str, Published]) -> tuple[bool, str]:
    proposal = latest_proposal(documents)
    if proposal is None:
        return False, "not closed pending: no proposed rule posted by as_of"
    for d in documents:
        end = _when(d.attributes.get("commentEndDate"))
        if end is not None and end > as_of:
            return False, f"not closed: {d.id}'s comment period ends {end.isoformat()}, after as_of"
    rule = final_rule(documents, proposal)
    if rule is None:
        return True, (f"closed, pending: every comment period ended by as_of and no Rule follows the proposal "
                      f"{proposal.id}, posted {proposal.attributes['postedDate']}: proposed, not final")  # fmt: skip
    in_effect, why = rule_status(rule, as_of, published)
    if in_effect:
        return False, f"not pending: the proposal {proposal.id} was made final by {why}"
    if in_effect is None:
        return True, (f"closed, status not known: the proposal {proposal.id} was made final by {why}; kept, since an "
                      "unknown effective date is not an effective one")  # fmt: skip
    return True, f"closed, pending: the proposal {proposal.id} was made final by {why}: final, not yet in effect"


def needs_effective(documents: Sequence[Entry]) -> str | None:
    """The Federal Register number whose effective date decides a docket's status, if any: its latest proposal's final
    rule's."""
    proposal = latest_proposal(documents)
    rule = final_rule(documents, proposal) if proposal is not None else None
    number = rule.attributes.get("frDocNum") if rule is not None else None
    return str(number) if number else None


def _status(asked: str, documents: Sequence[Entry], as_of: datetime, published: Mapping[str, Published]
            ) -> tuple[bool, str]:  # fmt: skip
    if asked == "open":
        return _open(documents, as_of)
    if asked == "closed_pending_final":
        return _pending(documents, as_of, published)
    is_open, why_open = _open(documents, as_of)
    if is_open:
        return True, why_open
    pending, why_pending = _pending(documents, as_of, published)
    return pending, why_pending if pending else f"{why_open}; {why_pending}"


def take_frame(criteria: FrameCriteria, client: RegulationsGov, store: Store, *, federal_register: FederalRegister,
               workers: int = WORKERS) -> Frame:  # fmt: skip
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
        # The effective date of each docket's latest proposal's final rule, wherever one follows it, whatever as_of
        # turns out to be (a superset of what the status tests read).
        numbers = {n for x in documents.values()
                   if (n := needs_effective([e for e in x.entries if not e.attributes.get("withdrawn")])) is not None}  # fmt: skip
        fr_responses, published = federal_register.documents(numbers)
        fr_artifacts = [commit(store, r, published_at=None, basis=LISTING_BASIS) for r in fr_responses]
        stated_in = {n: a.id for r, a in zip(fr_responses, fr_artifacts, strict=True) for n in asked_in(r.locator)}
        first_ended = max(p.retrieved_at for p in first)
        if criteria.as_of < first_ended:
            decided = _decide(criteria, searches, documents, evidence, criteria.as_of, first_ended, published)
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
    for d, (status, why, withdrawn) in _decide(criteria, searches, documents, evidence, as_of, ended, published).items():
        set_aside = f"; {withdrawn} withdrawn document(s) not counted as posted" if withdrawn else ""
        number = needs_effective([e for e in documents[d].entries if not e.attributes.get("withdrawn")])
        doc_artifacts = [a.id for a in evidence.pages(documents[d])] + ([stated_in[number]] if number else [])
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
    published: Mapping[str, Published],
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
        status, why = _status(criteria.comment_status, [e for e in posted if not e.attributes.get("withdrawn")], as_of,
                              published)  # fmt: skip
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
    """A docket's own measure, as its status test reads it: its latest Proposed Rule posted by as_of, or where it has
    none, the latest-posted document whose comment period runs at as_of. `documents` are the docket's documents posted
    by as_of, withdrawn ones set aside (docket_documents). An item is asked about the document it addresses (measures);
    this is its measure only where the source states none."""
    proposals = [d for d in documents if d.attributes.get("documentType") == "Proposed Rule"]
    pool = proposals or [d for d in documents if _runs(d, as_of)]
    return max(pool, key=lambda d: (instant(str(d.attributes["postedDate"])), d.id)) if pool else None


def _entries(store: Store, frame: Frame, docket: FrameDocket, published: dict[str, tuple[Published, str]]
             ) -> tuple[list[Entry], list[Artifact]]:  # fmt: skip
    """A frame docket's documents as its listing artifacts give them, and the Federal Register's statements it holds
    (into `published`: number -> (statement, its artifact))."""
    pages: list[Artifact] = []
    seen: dict[tuple[str, str, str], Entry] = {}
    for art_id in docket.artifacts:
        artifact = store.get("artifact", art_id, Artifact)
        if artifact is None:
            raise ValueError(f"frame {frame.id} names artifact {art_id}, which is not committed")
        if artifact.source == FEDERAL_REGISTER:
            for n, p in stated(artifact.locator, store.content(art_id)).items():
                published.setdefault(n, (p, art_id))
            continue
        if artifact.source != "regulations.gov":
            continue
        for e in json.loads(store.content(art_id)).get("data") or []:
            if e.get("type") == "documents":
                seen[identity(e)] = Entry(e, len(pages))
        pages.append(artifact)
    return list(seen.values()), pages


def _status_of(d: Entry, documents: Sequence[Entry], as_of: datetime, published: Mapping[str, Published]) -> str:
    kind = str(d.attributes.get("documentType"))
    withdrawn = " It is withdrawn." if d.attributes.get("withdrawn") else ""
    if kind == "Rule":
        in_effect, why = rule_status(d, as_of, published)
        return {True: "final and in effect: ", False: "final, not yet in effect: ", None: "final, effective date not "
                "known: "}[in_effect] + why + "." + withdrawn  # fmt: skip
    if kind != "Proposed Rule":
        if _runs(d, as_of):
            return f"a {kind} whose comment period runs to {d.attributes['commentEndDate']}; it states no rule itself." + withdrawn
        return f"a {kind}; it states no rule itself." + withdrawn
    rule = final_rule([e for e in documents if not e.attributes.get("withdrawn")], d)
    if rule is None:
        running = f" Its comment period runs to {d.attributes['commentEndDate']}." if _runs(d, as_of) else ""
        return f"proposed, not final: no final rule follows it in its docket by as_of.{running}" + withdrawn
    in_effect, why = rule_status(rule, as_of, published)
    return {True: "no longer pending: made final, in effect: ", False: "final, not yet in effect: made final by ",
            None: "made final, effective date not known: "}[in_effect] + why + "." + withdrawn  # fmt: skip


def _stated(d: Entry, status: str, as_of: datetime) -> str:
    a = d.attributes
    return (f'"{a.get("title")}" ({a.get("agencyId")} {a.get("documentType")} {d.id}, posted '
            f"{str(a.get('postedDate'))[:10]}). Its status as of {as_of.astimezone(EASTERN).date().isoformat()}: {status}")  # fmt: skip


def measures(store: Store, frame: Frame, items: Sequence[core.Item], *, regulations_gov: Callable[[], RegulationsGov],
             federal_register: Callable[[], FederalRegister]) -> dict[str, core.ItemMeasure]:  # fmt: skip
    """Each item's measure: the document it addresses, from its docket's documents as the frame listed them (withdrawn
    ones included, since a filing may address a document later withdrawn), or, where the frame's listing lacks it, from
    the document's own record, fetched and committed. Its status as of the frame's as_of comes from the documents posted
    by then and the Federal Register's effective dates. An item whose source states no document is asked about its
    docket's own measure (measure_document), and its statement says so."""
    dockets = {d.docket: d for d in frame.dockets}
    known: dict[str, tuple[Published, str]] = {}  # the Federal Register's statements the frame already holds
    docs_of: dict[str, tuple[list[Entry], list[Artifact]]] = {}
    by_id: dict[str, tuple[Entry, list[str]]] = {}
    wanted: dict[str, tuple[str, str | None]] = {}  # item -> (its docket, the document it addresses or None)
    for item in items:
        if item.container not in dockets:
            raise ValueError(f"item {item.id} is filed in {item.container}, which frame {frame.id} did not reach")
        if item.container not in docs_of:
            docs_of[item.container] = _entries(store, frame, dockets[item.container], known)
            for e in docs_of[item.container][0]:
                by_id.setdefault(e.id, (e, [a.id for a in docs_of[item.container][1]]))
        wanted[item.id] = (item.container, item.addresses)
    missing = sorted({doc for _, doc in wanted.values() if doc is not None and doc not in by_id})
    if missing:
        client = regulations_gov()
        try:
            for doc in missing:
                record = client.document(doc)
                art = commit(store, record, published_at=None, basis=LISTING_BASIS)
                if record.status == 404:
                    raise ValueError(f"a filing addresses {doc}, which regulations.gov no longer serves ({art.id})")
                by_id[doc] = (Entry(record.json()["data"], 0), [art.id])
        finally:
            client.close()
    as_of = frame.as_of

    def posted(docket: str) -> list[Entry]:
        entries, pages = docs_of[docket]
        return [e for e in entries
                if observable_at(listed_entry(pages[e.page], e), as_of, frame.retrieval_ended, frame.mode)]  # fmt: skip

    def chosen(item_id: str) -> tuple[Entry, str]:
        docket, doc = wanted[item_id]
        if doc is not None:
            return by_id[doc][0], docket
        own = measure_document([e for e in posted(docket) if not e.attributes.get("withdrawn")], as_of)
        if own is None:
            raise ValueError(f"item {item_id} names no document, and docket {docket} has no measure of its own")
        return own, docket

    def deciding_rule(d: Entry, docket: str) -> Entry | None:
        """The rule whose effective date decides a document's status: itself, if a rule; a proposal's final rule."""
        kind = d.attributes.get("documentType")
        if kind == "Rule":
            return d
        if kind == "Proposed Rule":
            return final_rule([e for e in posted(docket) if not e.attributes.get("withdrawn")], d)
        return None

    picks = {i: chosen(i) for i in wanted}
    rules = {i: deciding_rule(d, docket) for i, (d, docket) in picks.items()}
    numbers = {str(r.attributes["frDocNum"]) for r in rules.values() if r is not None and r.attributes.get("frDocNum")}
    published = {n: p for n, (p, _) in known.items()}
    stated_in = {n: a for n, (_, a) in known.items()}
    asked = sorted(numbers - set(known))
    if asked:
        fr = federal_register()
        try:
            responses, more = fr.documents(asked)
        finally:
            fr.close()
        published |= more
        stated_in |= {n: commit(store, r, published_at=None, basis=LISTING_BASIS).id for r in responses
                      for n in asked_in(r.locator)}  # fmt: skip
    out = {}
    for item_id, (d, docket) in picks.items():
        text = _stated(d, _status_of(d, posted(docket), as_of, published), as_of)
        if wanted[item_id][1] is None:
            text = f"{text} (The filing names no document it comments on; this is its docket's own measure.)"
        rule = rules[item_id]
        number = str(rule.attributes.get("frDocNum") or "") if rule is not None else ""
        arts = list(by_id[d.id][1])
        if number in stated_in:
            arts.append(stated_in[number])
        out[item_id] = core.ItemMeasure(item=item_id, document=d.id, statement=text, artifacts=list(dict.fromkeys(arts)))
    return out


def listed_entry(page: Artifact, e: Entry) -> Artifact:
    """A listed document as evidence posted at its postedDate (artifacts.listed)."""
    return listed(page, instant(str(e.attributes["postedDate"])), POSTED)


def frames(store: Store) -> list[Frame]:
    """The case's frozen frames, oldest first."""
    return store.all("frame", Frame)


def listed_comments(store: Store, frame: Frame, entries: Sequence[FrameItem]) -> dict[str, Mapping[str, Any]]:
    """Each entry's attributes as the frame's official listing of its docket's comments gives them (the listing pages
    the frame names as its artifacts): what a copy of the comment's record from elsewhere must agree with to be taken
    as current (requisite/evidence/mirrulations.py)."""
    dockets = {d.docket: d for d in frame.dockets}
    wanted = {e.id for e in entries}
    out: dict[str, Mapping[str, Any]] = {}
    for docket in sorted({e.docket for e in entries}):
        for art_id in dockets[docket].artifacts:
            artifact = store.get("artifact", art_id, Artifact)
            if artifact is None or artifact.source != "regulations.gov":
                continue
            for e in json.loads(store.content(art_id)).get("data") or []:
                if e.get("type") == "comments" and e.get("id") in wanted:
                    out[str(e["id"])] = e.get("attributes") or {}
    return out


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


def frame_gate(client: Callable[[], RegulationsGov], federal_register: Callable[[], FederalRegister] = FederalRegister
               ) -> Callable[[Store, core.Proposal], GateResult]:  # fmt: skip
    """The tests gate of a frame proposal (core.REQUIRED_GATES: "frame (its criteria run and freeze a frame)"): the
    criteria run and freeze a frame; it passes when the frame holds an item to sample."""

    def check(store: Store, proposal: core.Proposal) -> GateResult:
        criteria = FrameCriteria.model_validate_json(core.canonical_json(dict(proposal.payload)))
        regulations_gov, fr = client(), federal_register()
        try:
            frame = take_frame(criteria, regulations_gov, store, federal_register=fr)
        finally:
            regulations_gov.close()
            fr.close()
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
