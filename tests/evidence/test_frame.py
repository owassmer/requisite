"""A frame from its criteria, frozen (spec §7.2 step 1): every test taken as of as_of with only what was posted by then
(spec I3), the comment status and count of each docket reached, the comments of the kept dockets listed by as_of, a
withdrawn comment kept and marked, and every count recomputing from the artifacts the frame names."""

import copy
import json
from datetime import UTC, date, datetime

import pytest
from fake_api import DOCKET, FakeAPI, comment_entry, fixture_world, proposal_0001

from requisite import core
from requisite.core import Frame, FrameCriteria
from requisite.evidence.frame import entries, take_frame
from requisite.evidence.regulations_gov import instant


def criteria(as_of: datetime, status: str = "either", min_comments: int = 2) -> FrameCriteria:
    return FrameCriteria(
        id="frame", version=1, objective=core.Ref(id="objective", version=1), source="regulations.gov", as_of=as_of,
        search_terms=["Airworthiness Directives"], document_types=["Proposed Rule", "Rule"],
        posted_from=date(2026, 4, 1), posted_to=None, agencies=["FAA"], comment_status=status,
        min_comments=min_comments,
    )  # fmt: skip


def docket(frame: Frame, name: str):
    (d,) = [d for d in frame.dockets if d.docket == name]
    return d


def world() -> FakeAPI:
    api = fixture_world(FakeAPI())
    api.add_comment(DOCKET, comment_entry(f"{DOCKET}-0002", "2026-05-08T04:00:00Z", "2026-05-20T11:00:00Z", withdrawn=True))
    later = copy.deepcopy(proposal_0001())  # a proposal in another docket, posted the day after the as_of below
    later["id"], later["attributes"]["docketId"] = "FAA-2026-9999-0001", "FAA-2026-9999"
    later["attributes"].update(postedDate="2026-05-13T04:00:00Z", lastModifiedDate="2026-05-13T15:00:00Z")
    api.add_document(later)
    api.add_comment("FAA-2026-9999", comment_entry("FAA-2026-9999-0003", "2026-05-13T04:00:00Z", "2026-05-13T16:00:00Z"))
    return api


def test_a_frame_holds_only_what_was_posted_by_its_as_of(live, make_client, store):
    # 23:30 in New York on May 11: half an hour before comment 0004 and the other docket's proposal were posted (their
    # postedDate is the next midnight there); the API's day filter serves them, and observable_at excludes them.
    as_of = datetime(2026, 5, 12, 3, 30, tzinfo=UTC)
    api = world()
    frame = take_frame(criteria(as_of, "open", min_comments=2), make_client(api, live / "cassettes", live), store)
    assert frame.mode == "retrospective_reconstruction" and frame.as_of == as_of
    assert frame.retrieval_started <= frame.retrieval_ended
    assert [(s.term, s.documents) for s in frame.searches] == [("Airworthiness Directives", 2), ("Airworthiness Directives", 0)]
    assert frame.searches[0].request == (
        "https://api.regulations.gov/v4/documents?filter[agencyId]=FAA&filter[documentType]=Proposed%20Rule"
        "&filter[postedDate][ge]=2026-04-01&filter[postedDate][le]=2026-05-13&filter[searchTerm]=Airworthiness%20Directives"
        "&page[size]=250&sort=lastModifiedDate%2CdocumentId"
    ).replace("%20", "+")
    assert [d.docket for d in frame.dockets] == [DOCKET]  # FAA-2026-9999 was reached only by a later document
    kept = docket(frame, DOCKET)
    assert kept.kept and kept.comments == 2
    assert kept.reason.startswith(f"open: {DOCKET}-0001's comment period ends 2026-06-05T03:59:59+00:00, after as_of")
    assert [(i.id, i.withdrawn) for i in frame.items] == [(f"{DOCKET}-0003", False), (f"{DOCKET}-0002", True)]
    served = {e["id"] for a in kept.artifacts for e in json.loads(store.content(a))["data"]}
    assert f"{DOCKET}-0004" in served  # in the listing the day filter served, excluded by as_of
    assert store.get("frame", frame.id, Frame) == frame
    assert entries(store, [f"{DOCKET}-0002"]) == [frame.items[1]]
    with pytest.raises(ValueError, match="in no frame of this case"):
        entries(store, [f"{DOCKET}-0004"])


def test_a_final_rule_posted_after_as_of_is_not_known_then(live, make_client, store):
    pending = take_frame(criteria(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final"),
                         make_client(world(), live / "a", live), store)  # fmt: skip
    d = docket(pending, DOCKET)
    assert d.kept and d.reason.startswith(f"closed pending a final rule: every comment period ended by as_of and no Rule "
                                          f"follows the proposal {DOCKET}-0001")  # fmt: skip
    assert [i.id for i in pending.items] == [f"{DOCKET}-{n}" for n in ("0003", "0002", "0004", "0005")]

    final = take_frame(criteria(datetime(2026, 10, 15, 12, tzinfo=UTC), "closed_pending_final"),
                       make_client(world(), live / "b", live), store)  # fmt: skip
    d = docket(final, DOCKET)
    assert not d.kept and d.comments == 4
    assert d.reason.startswith(f"not pending a final rule: {DOCKET}-0006 (Rule) was posted 2026-10-01T04:00:00Z")
    assert docket(final, "FAA-2026-3487").reason.startswith("not closed pending a final rule: no proposed rule posted")
    assert docket(final, "FAA-2026-11796").reason.startswith("not closed: FAA-2026-11796-0001's comment period ends")
    assert all(not d.kept for d in final.dockets) and final.items == ()
    assert "0 public comments posted by as_of, fewer than the 2 asked" in docket(final, "FAA-2026-3487").reason


def test_a_frame_asked_for_a_later_instant_is_prospective_as_of_the_end_of_its_retrieval(live, make_client, store):
    frame = take_frame(criteria(datetime(2100, 1, 1, tzinfo=UTC)), make_client(world(), live / "c", live), store)
    assert frame.mode == "prospective" and frame.as_of == frame.retrieval_ended
    assert frame.retrieval_ended <= datetime.now(UTC)
    # Its proposal's period ended in June and a Rule followed in October: neither open nor pending, at any later date.
    assert not docket(frame, DOCKET).kept


def test_every_count_of_a_frame_recomputes_from_the_artifacts_it_names(live, make_client, store):
    frame = take_frame(criteria(datetime(2026, 10, 15, 12, tzinfo=UTC)), make_client(world(), live / "d", live), store)
    for s in frame.searches:
        found = {e["id"] for a in s.artifacts for e in json.loads(store.content(a))["data"]}
        assert len(found) == s.documents
    for d in frame.dockets:
        listed = {e["id"]: e for a in d.artifacts for e in json.loads(store.content(a))["data"] if e["type"] == "comments"}
        assert sum(instant(e["attributes"]["postedDate"]) <= frame.as_of for e in listed.values()) == d.comments
    stamps = [store.get("artifact", a, core.Artifact).observed_at for s in frame.searches for a in s.artifacts]
    assert min(stamps) >= frame.retrieval_started and max(stamps) <= frame.retrieval_ended
