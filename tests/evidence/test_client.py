"""The regulations.gov client against the fake API: pagination and its continuation past 20 pages, pacing by the stated
remaining requests, a 429 waited out, transient failures retried and lasting ones raised, and the key kept to its
header."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fake_api import DOCKET, KEY, PDF_0005, FakeAPI, FakeTime, comment_entry, fixture_world

from requisite.evidence.regulations_gov import MAX_PAGES, PAGE_SIZE, Pacer, RegulationsGovError, eastern_seam


def docket_of(api: FakeAPI, n: int, *, same_second: int = 1) -> list[str]:
    """n comments in one docket, each `same_second` of them sharing a lastModifiedDate."""
    start = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)
    ids = []
    for i in range(n):
        modified = (start + timedelta(seconds=i // same_second)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cid = f"X-2026-0001-{i:05d}"
        api.add_comment("X-2026-0001", comment_entry(cid, "2026-03-02T05:00:00Z", modified))
        ids.append(cid)
    return ids


def requests_to(api: FakeAPI, path: str) -> list[dict[str, str]]:
    return [dict(r.url.params) for r in api.requests if r.url.path == f"/v4/{path}"]


def test_a_listing_pages_through_every_result_at_the_apis_largest_page(live, make_client):
    api = FakeAPI()
    ids = docket_of(api, 600)
    listing = make_client(api, live / "cassettes", live).comments({"filter[docketId]": "X-2026-0001"})
    assert [e.id for e in listing.entries] == ids
    sent = requests_to(api, "comments")
    assert [p["page[number]"] for p in sent] == ["1", "2", "3"]
    assert {p["page[size]"] for p in sent} == {str(PAGE_SIZE)} and {p["sort"] for p in sent} == {"lastModifiedDate,documentId"}
    assert len(listing.pages) == 3 and all(p.status == 200 for p in listing.pages)
    assert listing.request == ("https://api.regulations.gov/v4/comments?filter[docketId]=X-2026-0001&page[size]=250"
                               "&sort=lastModifiedDate%2CdocumentId")  # fmt: skip


def test_a_listing_past_twenty_pages_continues_from_the_last_lastModifiedDate_and_drops_the_seams_repeats(live, make_client):
    api = FakeAPI()
    ids = docket_of(api, 5300, same_second=7)  # the 5000th result shares its second with the ones after it
    listing = make_client(api, live / "cassettes", live).comments({"filter[docketId]": "X-2026-0001"})
    assert sorted(e.id for e in listing.entries) == ids  # every result, each once
    sent = requests_to(api, "comments")
    assert len(sent) == MAX_PAGES + 2 and len(listing.pages) == len(sent)
    assert all("filter[lastModifiedDate][ge]" not in p for p in sent[:MAX_PAGES])
    seam = {p["filter[lastModifiedDate][ge]"] for p in sent[MAX_PAGES:]}
    # The 5000th result was modified 714 seconds after 14:00Z on March 2: 09:11:54 in Eastern (standard) time.
    assert seam == {"2026-03-02 09:11:54"}
    assert [p["page[number]"] for p in sent[MAX_PAGES:]] == ["1", "2"]


def test_a_listing_disturbed_by_a_change_to_what_it_already_read_is_listed_again(live, make_client):
    api = FakeAPI()
    ids = docket_of(api, 600)

    def edit_the_first_after_page_one(request):
        if request.url.params.get("page[number]") == "1" and len(api.requests) == 1:
            _, first = api.comments[ids[0]]
            first["attributes"]["lastModifiedDate"] = "2026-03-03T09:00:00Z"  # it moves last; the rest move up a place

    api.after.append(edit_the_first_after_page_one)
    listing = make_client(api, live / "cassettes", live).comments({"filter[docketId]": "X-2026-0001"})
    first_pass = {e["id"] for page in listing.pages[:3] for e in page.json()["data"]}
    assert ids[250] not in first_pass  # the shift hid it from the first pass
    assert sorted(e.id for e in listing.entries) == ids  # and the second found it
    assert len(requests_to(api, "comments")) == 6 and len(listing.pages) == 6  # two passes of three pages
    kept = [json.loads(f.read_text())["request"] for f in (live / "cassettes" / "regulations_gov").glob("*.json")]
    assert sum(1 for k in kept if k.get("pass") == 2) == 3  # the second pass is its own requests


def test_a_seam_in_the_autumn_hour_restarts_early_enough_for_either_reading():
    # 2026-11-01 01:30 occurs twice in New York: first at 05:30Z (EDT), then at 06:30Z (EST).
    assert eastern_seam("2026-11-01T05:30:00Z") == "2026-11-01 00:30:00"
    assert eastern_seam("2026-11-01T06:30:00Z") == "2026-11-01 01:30:00"
    assert eastern_seam("2020-08-10T15:58:52Z") == "2020-08-10 11:58:52"  # the API documentation's own example


def test_a_long_read_waits_for_the_hour_to_free_a_slot_and_never_runs_the_key_dry(live, make_client):
    time = FakeTime()
    api = fixture_world(FakeAPI(time=time, limit=15))
    c = make_client(api, live / "cassettes", live, time)
    for i in range(8):
        c.api(f"dockets/{DOCKET}", {"n": i})  # eight distinct requests
    assert 429 not in api.served and len(api.requests) == 8
    # Fifteen an hour with ten kept in reserve: five go at once (one second each); the sixth waits until the first leaves
    # the hour, the seventh until the second does; by then the API counts two fewer, and the eighth goes at once.
    assert time.slept == [3595.0, 1.0]


def test_a_429_is_waited_out_and_retried_never_dropped(live, make_client):
    time = FakeTime()
    api = fixture_world(FakeAPI(time=time))
    api.script[f"comments/{DOCKET}-0005"] = [429]
    record = make_client(api, live / "cassettes", live, time).comment(f"{DOCKET}-0005")
    assert record.status == 200 and record.json()["data"]["id"] == f"{DOCKET}-0005"
    assert api.served == [429, 200]
    # None of the key's spending is this run's (the refused request spent nothing): it probes again after a minute.
    assert time.slept == [60.0]

    api = fixture_world(FakeAPI(time=FakeTime(), retry_after="30"))
    api.script[f"comments/{DOCKET}-0004"] = [429, 429]
    make_client(api, live / "c2", live).comment(f"{DOCKET}-0004")
    assert api.time.slept == [30.0, 30.0] and api.served == [429, 429, 200]


def test_a_transient_failure_is_retried_and_a_lasting_one_raises_with_nothing_recorded(live, make_client):
    api = fixture_world(FakeAPI())
    api.script[f"comments/{DOCKET}-0004"] = [503, 502]
    assert make_client(api, live / "a", live).comment(f"{DOCKET}-0004").status == 200
    api.script[f"comments/{DOCKET}-0003"] = [503] * 50
    with pytest.raises(RegulationsGovError, match="503 after 6 retries"):
        make_client(api, live / "b", live).comment(f"{DOCKET}-0003")
    api.script[f"dockets/{DOCKET}"] = [400]
    with pytest.raises(RegulationsGovError, match="HTTP 400"):
        make_client(api, live / "c", live).docket(DOCKET)
    assert not list((live / "b").rglob("*.json")) and not list((live / "c").rglob("*.json"))


def test_the_key_travels_only_in_its_header_and_a_replay_never_asks_for_it(live, make_client, monkeypatch):
    api = fixture_world(FakeAPI())
    c = make_client(api, live / "cassettes", live)
    c.comment(f"{DOCKET}-0005")
    pdf = c.download(PDF_0005)
    assert pdf.status == 200 and pdf.content[:5] == b"%PDF-"
    assert api.requests[0].headers["x-api-key"] == KEY and "x-api-key" not in api.requests[1].headers
    for f in (live / "cassettes").rglob("*.json"):
        assert KEY not in f.read_text()
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    again = make_client(None, live / "cassettes", live)  # its key raises if asked
    assert again.comment(f"{DOCKET}-0005").content == c.comment(f"{DOCKET}-0005").content
    assert again.download(PDF_0005).content == pdf.content


def test_the_pacer_lets_requests_go_while_the_reserve_holds_beyond_those_in_flight():
    time = FakeTime()
    pacer = Pacer(time.clock, time.sleep, reserve=2)
    for _ in range(3):
        pacer.acquire()  # nothing stated yet: requests go
    pacer.settle(5)  # 5 remain and 2 are in flight: 3 beyond the reserve
    pacer.acquire()  # goes: now 3 in flight, 2 beyond the reserve
    assert time.slept == []
    pacer.acquire()  # at the reserve: it waits for the oldest of its requests to leave the hour
    assert time.slept == [3600.0]
