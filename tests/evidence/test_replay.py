"""The frame and its read replay from cassettes alone: recorded once over the fixture transport into a fresh cassette
directory (as the coordinator's live run records), then run again with the transport removed, into a fresh case, with no
network and no OCR tool, giving the same frame, artifacts, extractions and items; and the flow's CLI steps do the same,
each run a second time reading everything from cassettes."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fake_api import DOCKET, FakeAPI, comment_entry, fixture_world
from test_frame import criteria

from requisite import cassettes, cli, core
from requisite.core import Artifact, Item
from requisite.evidence import extract
from requisite.evidence.frame import frame_gate, take_frame
from requisite.evidence.items import read_items
from requisite.evidence.regulations_gov import RegulationsGov
from requisite.store import Store

CRITERIA = criteria(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final")


def record(live, make_client, scan):
    api = fixture_world(FakeAPI(), scanned_0003=scan)
    api.add_comment(DOCKET, comment_entry(f"{DOCKET}-0002", "2026-05-08T04:00:00Z", "2026-05-20T11:00:00Z", withdrawn=True))
    client = make_client(api, live / "cassettes", live)
    store = Store(live / "recorded", case="wide")
    frame = take_frame(CRITERIA, client, store)
    items = read_items(client, store, frame.items)
    return frame, items, store


@pytest.fixture
def offline(monkeypatch):
    """Replay only, the network unreachable and the OCR tools unusable: anything not in a cassette fails loudly."""

    def refuse(*args, **kwargs):
        raise AssertionError("a replay reached for the network or a local tool")

    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(extract, "_run", refuse)
    monkeypatch.setattr(extract, "read", refuse)


def test_a_frame_and_its_items_replay_from_cassettes_with_the_transport_removed(live, make_client, scan, monkeypatch, request):
    frame, items, recorded = record(live, make_client, scan)
    assert frame.mode == "retrospective_reconstruction" and len(items) == 4
    assert [i.withdrawn for i in frame.items] == [False, True, False, False]
    request.getfixturevalue("offline")
    fresh = Store(live / "fresh", case="wide")
    client = make_client(None, live / "cassettes", live)  # no transport of ours: a request would go to the network
    again = take_frame(CRITERIA, client, fresh)
    assert again == fresh.get("frame", again.id, type(again)) == frame
    assert read_items(client, fresh, again.items) == items
    assert {a.id: a for a in fresh.all("artifact", Artifact)} == {a.id: a for a in recorded.all("artifact", Artifact)}
    for item in items:
        for art, x in zip(item.artifacts, item.extractions, strict=False):
            assert fresh.extraction(art, x) == recorded.extraction(art, x)
    assert fresh.all("item", Item) == recorded.all("item", Item)
    # The replayed frame's times are the recording's: its retrieval ended when the last response was recorded.
    assert again.retrieval_ended - again.retrieval_started < timedelta(minutes=5)


def test_the_flow_freezes_its_frame_through_the_gate_and_reads_its_sample_from_cassettes_the_same_each_time(
    live, make_client, scan, monkeypatch, capsys, request
):
    # The frame is taken only by a frame proposal's tests gate, and the sample read is the design's draw: the CLI has no
    # command that freezes hand-written criteria or reads hand-picked ids (decisions, "Discovery finds what it analyzes").
    frame, items, _ = record(live, make_client, scan)
    request.getfixturevalue("offline")
    monkeypatch.setattr(cassettes, "CASSETTES", live / "cassettes")
    runs = []
    for n in range(2):
        store = Store(live / f"cli-case-{n}", case="wide")
        store.register_gate("frame", "tests", frame_gate(RegulationsGov))
        proposal = store.submit_proposal(core.Proposal(
            id="frame-fixture", case="wide", kind="frame", payload=CRITERIA.model_dump(mode="json"), author="fixture",
            rationale="the fixture's criteria", status="proposed",
        ))  # fmt: skip
        assert proposal.status == "active"
        case = ["--case-dir", str(live / f"cli-case-{n}"), "--case", "wide"]
        assert cli.main(["wide", "sample", *case]) == 0
        assert cli.main(["wide", "read", *case]) == 0
        runs.append(capsys.readouterr().out)
        assert Store(live / f"cli-case-{n}", case="wide").all("item", Item) == items
    assert runs[0] == runs[1]
    assert f"design two_phase.{frame.id}  phase 1  seed " in runs[0]
    assert "  stratum under_100: 4 of 4 items, each with probability 1" in runs[0]  # a frame this small is a census
    assert runs[0].rstrip().endswith("4 items read")
    assert f"{DOCKET}-0005  {DOCKET}  2026-05-19T04:00:00+00:00  United Airlines, Inc  1 file(s)" in runs[0]
    assert "unreadable: the comment's record: regulations.gov answered 404 for it" in runs[0]  # 0002 is listed, not served
