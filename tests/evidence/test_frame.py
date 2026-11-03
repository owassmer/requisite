"""A frame from its criteria, frozen (spec §7.2 step 1): every test taken as of as_of with only what was posted by then
(spec I3), the comment status of each docket reached and the count of each its status does not drop, the comments of the kept dockets listed by as_of, a
withdrawn comment kept and marked, and every count recomputing from the artifacts the frame names."""

import copy
import json
from datetime import UTC, date, datetime

import pytest
from fake_api import DOCKET, FakeAPI, FakeFederalRegister, comment_entry, fixture_world, proposal_0001

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


FINAL_0006 = "2026-20120"  # the Federal Register number of the docket's final rule, FAA-2026-2730-0006 (posted Oct 1)


def register(effective: str | None = "2026-10-10") -> FakeFederalRegister:
    """The Federal Register as these tests see it: the docket's final rule takes effect on `effective`."""
    return FakeFederalRegister(effective={FINAL_0006: effective})


def take(as_of, status, root, live, make_client, make_fr, store, *, effective="2026-10-10", min_comments=2, api=None):
    return take_frame(criteria(as_of, status, min_comments=min_comments), make_client(api or world(), live / root, live),
                      store, federal_register=make_fr(register(effective), live / root, live))  # fmt: skip


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


def test_a_frame_holds_only_what_was_posted_by_its_as_of(live, make_client, make_fr, store):
    # 23:30 in New York on May 11: half an hour before comment 0004 and the other docket's proposal were posted (their
    # postedDate is the next midnight there); the API's day filter serves them, and observable_at excludes them.
    as_of = datetime(2026, 5, 12, 3, 30, tzinfo=UTC)
    api = world()
    frame = take(as_of, "open", "cassettes", live, make_client, make_fr, store, api=api)
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


def test_a_final_rule_posted_after_as_of_is_not_known_then(live, make_client, make_fr, store):
    pending = take(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final", "a", live, make_client, make_fr, store)
    d = docket(pending, DOCKET)
    assert d.kept and d.reason.startswith(f"closed, pending: every comment period ended by as_of and no Rule follows "
                                          f"the proposal {DOCKET}-0001")  # fmt: skip
    assert [i.id for i in pending.items] == [f"{DOCKET}-{n}" for n in ("0003", "0002", "0004", "0005")]

    final = take(datetime(2026, 10, 15, 12, tzinfo=UTC), "closed_pending_final", "b", live, make_client, make_fr, store)
    d = docket(final, DOCKET)
    assert not d.kept and d.comments is None  # its status drops it before its comments are listed: no count is stated
    assert d.reason.endswith("its comments were not listed, since its status drops it")
    assert d.reason.startswith(f"not pending: the proposal {DOCKET}-0001 was made final by {DOCKET}-0006 (Rule, posted "
                               f"2026-10-01) took effect 2026-10-10 (Federal Register {FINAL_0006}), by as_of")  # fmt: skip
    assert docket(final, "FAA-2026-3487").reason.startswith("not closed pending: no proposed rule posted")
    assert docket(final, "FAA-2026-11796").reason.startswith("not closed: FAA-2026-11796-0001's comment period ends")
    assert all(not d.kept for d in final.dockets) and final.items == ()

    few = take(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final", "e", live, make_client, make_fr, store,
               min_comments=5)  # fmt: skip
    d = docket(few, DOCKET)
    assert not d.kept and d.comments == 4 and "4 public comments posted by as_of, fewer than the 5 asked" in d.reason


def test_a_final_rule_not_yet_in_effect_is_still_pending_and_one_whose_date_is_not_stated_is_not_known(
    live, make_client, make_fr, store
):
    # The objective's events include a measure that is final but not yet in effect (external evaluation, November 2:
    # every docket whose proposal a final rule followed had been dropped). The limit is the effective day in Washington:
    # as_of at 08:00 on October 15 there is October 15.
    as_of = datetime(2026, 10, 15, 12, tzinfo=UTC)
    later = take(as_of, "closed_pending_final", "f", live, make_client, make_fr, store, effective="2026-10-16")
    d = docket(later, DOCKET)
    assert d.kept and d.reason.startswith(
        f"closed, pending: the proposal {DOCKET}-0001 was made final by {DOCKET}-0006 (Rule, posted 2026-10-01) takes "
        f"effect 2026-10-16 (Federal Register {FINAL_0006}), after as_of: final, not yet in effect")  # fmt: skip
    assert {i.id for i in later.items} >= {f"{DOCKET}-0003", f"{DOCKET}-0005"}
    # Its stated date is an artifact the docket names, so its status recomputes from what was retrieved.
    assert any(store.get("artifact", a, core.Artifact).source == "federalregister.gov" for a in d.artifacts)
    same_day = take(as_of, "closed_pending_final", "g", live, make_client, make_fr, store, effective="2026-10-15")
    assert not docket(same_day, DOCKET).kept  # in effect on as_of's own day
    unstated = take(as_of, "closed_pending_final", "h", live, make_client, make_fr, store, effective=None)
    d = docket(unstated, DOCKET)
    assert d.kept and "states no effective date" in d.reason and "an unknown effective date is not an effective one" in d.reason


def test_a_frame_asked_for_a_later_instant_is_prospective_as_of_the_end_of_its_retrieval(live, make_client, make_fr, store):
    frame = take(datetime(2100, 1, 1, tzinfo=UTC), "either", "c", live, make_client, make_fr, store)
    assert frame.mode == "prospective" and frame.as_of == frame.retrieval_ended
    assert frame.retrieval_ended <= datetime.now(UTC)
    # Its proposal's period ended in June and its final rule took effect in October: neither open nor pending later.
    assert not docket(frame, DOCKET).kept
    # Its as_of was not fixed before the comments were listed, so every docket reached was counted.
    assert all(d.comments is not None for d in frame.dockets)


def test_every_count_of_a_frame_recomputes_from_the_artifacts_it_names(live, make_client, make_fr, store):
    frame = take(datetime(2026, 10, 15, 12, tzinfo=UTC), "either", "d", live, make_client, make_fr, store)
    for s in frame.searches:
        found = {e["id"] for a in s.artifacts for e in json.loads(store.content(a))["data"]}
        assert len(found) == s.documents
    for d in frame.dockets:
        listed = {e["id"]: e for a in d.artifacts for e in json.loads(store.content(a)).get("data") or []
                  if e["type"] == "comments"}  # fmt: skip
        if d.comments is None:  # dropped by its status before its comments were listed: none were retrieved
            assert not d.kept and not listed
            continue
        assert sum(instant(e["attributes"]["postedDate"]) <= frame.as_of for e in listed.values()) == d.comments
    stamps = [store.get("artifact", a, core.Artifact).observed_at for s in frame.searches for a in s.artifacts]
    assert min(stamps) >= frame.retrieval_started and max(stamps) <= frame.retrieval_ended


def test_a_comment_id_listed_for_two_records_is_one_item_its_record_not_withdrawn(live, make_client, make_fr, store):
    # regulations.gov listed CMS-2025-0058-0103 for two records, one withdrawn (November 2): a frame item is a comment,
    # so the frame holds it once, as its record not withdrawn, and counts it once.
    api = world()
    again = comment_entry(f"{DOCKET}-0004", "2026-05-08T04:00:00Z", "2026-09-24T01:05:10Z", withdrawn=True)
    api.add_comment(DOCKET, again, key=f"{DOCKET}-0004@withdrawn")
    one = take(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final", "dup", live, make_client, make_fr, store,
               api=api)  # fmt: skip
    ids = [i.id for i in one.items]
    assert ids.count(f"{DOCKET}-0004") == 1
    (item,) = [i for i in one.items if i.id == f"{DOCKET}-0004"]
    assert not item.withdrawn
    d = docket(one, DOCKET)
    assert d.comments == len(ids) and f"listed for two or more records ({DOCKET}-0004)" in d.reason
