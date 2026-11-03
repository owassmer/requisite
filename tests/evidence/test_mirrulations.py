"""A comment's record read from the Mirrulations mirror where its copy is current against the frame's official listing,
and from regulations.gov's API otherwise (requisite/evidence/mirrulations.py; decisions, "Sources")."""

import copy
from dataclasses import dataclass, field
from typing import Any

import httpx
from fake_api import DOCKET, FakeAPI, fixture_world, load
from test_items import entry

from requisite.core import Artifact
from requisite.evidence import items
from requisite.evidence.mirrulations import SOURCE, Mirrulations, comment_key


@dataclass
class FakeMirror:
    """The bucket: each key it holds answered with its bytes, any other with 403, as S3 answers a public bucket."""

    held: dict[str, bytes] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "mirrulations.s3.amazonaws.com"
        self.requests.append(request)
        key = request.url.path.lstrip("/")
        if key in self.held:
            return httpx.Response(200, content=self.held[key], headers={"content-type": "application/json"})
        return httpx.Response(403, content=b"<Error><Code>AccessDenied</Code></Error>")


def mirrored(record: dict[str, Any]) -> bytes:
    """A record as the mirror stores it: the API's detail response without its links block."""
    body = copy.deepcopy(record)
    body.pop("links", None)
    import json

    return json.dumps(body).encode()


def test_a_current_copy_is_read_from_the_mirror_and_a_stale_or_missing_one_from_the_api(live, make_client, store, scan):
    api = fixture_world(FakeAPI(), scanned_0003=scan)
    official = {e["id"]: e["attributes"] for e in load("comments_list.json")["body"]["data"]}
    details = {load(f"comment_{i}.json")["body"]["data"]["id"]: load(f"comment_{i}.json")["body"] for i in range(3)}
    current, stale, missing = f"{DOCKET}-0005", f"{DOCKET}-0004", f"{DOCKET}-0003"
    # This test's premise: the official listing and the API's record agree on every field they share.
    for cid in (current, stale):
        record = details[cid]["data"]["attributes"]
        assert all(record[k] == v for k, v in official[cid].items() if k in record), cid
    old = copy.deepcopy(details[stale])
    old["data"]["attributes"]["modifyDate"] = "2026-01-01T00:00:00Z"  # an earlier copy than the one listed
    bucket = FakeMirror({comment_key(current): mirrored(details[current]), comment_key(stale): mirrored(old)})
    mirror = Mirrulations(transport=bucket.transport(), root=live / "cassettes")
    client = make_client(api, live / "cassettes", live)

    read = items.read_items(client, store, [entry("0005"), entry("0004"), entry("0003")], workers=1, mirror=mirror,
                            official=official)  # fmt: skip
    asked = {r.url.path.rsplit("/", 1)[-1] for r in api.requests if "/v4/comments/" in r.url.path}
    assert asked == {stale, missing}  # the current copy spent no request of the key's hour
    assert len(bucket.requests) == 3
    by = {i.id: i for i in read}
    record = store.get("artifact", by[current].artifacts[0], Artifact)
    assert record is not None and record.source == SOURCE and record.published_at_basis == items.MIRROR_BASIS
    for cid in (stale, missing):
        api_record = store.get("artifact", by[cid].artifacts[0], Artifact)
        assert api_record is not None and api_record.source == "regulations.gov"
    # The mirror's copy reads as the API's record does: the same comment text and the same files.
    alone = items.read_items(client, type(store)(live / "alone", case="alone"), [entry("0005")], workers=1)[0]
    assert by[current].text.split("\n", 1)[1] == alone.text.split("\n", 1)[1]
    assert by[current].artifacts[1:] == alone.artifacts[1:]


def test_a_copy_is_current_only_where_it_carries_and_agrees_on_what_the_listing_gives():
    import json
    from datetime import UTC, datetime

    from requisite.core import FrameItem

    listed = {"postedDate": "2026-11-01T04:00:00Z", "withdrawn": False, "lastModifiedDate": "m", "title": "t"}
    record = {"docketId": "D", "postedDate": "2026-11-01T04:00:00Z", "withdrawn": False, "modifyDate": "m", "title": "t"}
    one = FrameItem(id="D-1", docket="D", posted_at=datetime(2026, 11, 1, 4, tzinfo=UTC), withdrawn=False)

    def copy_of(**attributes: Any) -> items.Fetched:
        body = {"data": {"id": "D-1", "attributes": {**record, **attributes}}}
        return items.Fetched(source=SOURCE, locator="x", status=200, media_type="application/json",
                             content=json.dumps(body).encode(), retrieved_at=datetime(2026, 11, 2, tzinfo=UTC))  # fmt: skip

    assert items.current(copy_of(), one, listed)
    assert not items.current(copy_of(modifyDate="earlier"), one, listed)  # an older copy
    assert not items.current(copy_of(title="another"), one, listed)
    assert not items.current(copy_of(docketId="E"), one, listed)
    assert not items.current(copy_of(), one.model_copy(update={"id": "D-2"}), listed)
    assert not items.current(copy_of(), one, {k: v for k, v in listed.items() if k != "lastModifiedDate"})
