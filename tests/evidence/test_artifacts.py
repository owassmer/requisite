"""Retrieved responses as artifacts (spec I2, I3): the same bytes retrieved twice are one artifact, and a record
re-retrieved with any field changed is a new artifact linked to its predecessor, its text unchanged by value."""

from fake_api import DOCKET, FakeAPI, fixture_world

from requisite.core import Artifact
from requisite.core.evidence import artifact_id
from requisite.evidence import extract
from requisite.evidence.artifacts import commit
from requisite.evidence.regulations_gov import instant

ID = f"{DOCKET}-0005"


def retrieve(api, root, tmp, make_client, store) -> Artifact:
    record = make_client(api, root, tmp).comment(ID)
    posted = instant(record.json()["data"]["attributes"]["postedDate"])
    return commit(store, record, published_at=posted, basis="the comment's postedDate (regulations.gov)")


def test_the_same_bytes_retrieved_twice_are_one_artifact_observed_at_its_first_retrieval(live, make_client, store):
    api = fixture_world(FakeAPI())
    first = retrieve(api, live / "run1", live, make_client, store)
    again = retrieve(api, live / "run2", live, make_client, store)  # a second live retrieval, its own cassettes
    assert len(api.requests) == 2
    assert again == first and first.predecessor is None
    assert [a.id for a in store.artifacts_at(first.locator)] == [first.id]
    assert first.locator == f"https://api.regulations.gov/v4/comments/{ID}?include=attachments"
    assert first.published_at == instant("2026-05-19T04:00:00Z") and first.availability == "retrieved"
    assert first.id == artifact_id(store.content(first.id))


def test_a_record_retrieved_with_a_field_changed_is_a_new_artifact_linked_to_its_predecessor(live, make_client, store):
    api = fixture_world(FakeAPI())
    first = retrieve(api, live / "run1", live, make_client, store)
    api.details[f"comments/{ID}"]["data"]["attributes"]["modifyDate"] = "2026-06-02T10:00:00Z"  # metadata only
    second = retrieve(api, live / "run2", live, make_client, store)
    assert second.id != first.id and second.predecessor == first.id
    assert [a.id for a in store.artifacts_at(first.locator)] == [first.id, second.id]
    # What derives from the text compares by value: the comment's text did not change.
    texts = {extract.read("comment", store.content(a)).text for a in (first.id, second.id)}
    assert len(texts) == 1
