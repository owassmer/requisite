"""Items read whole (decisions: an item's text is everything its submitter filed): the comment and then each file in
the order filed, each part headed by the artifact it came from; what could not be read stated, never dropped."""

import copy
from datetime import UTC, datetime

import pypdf
from fake_api import DOCKET, FakeAPI, fixture_pdf, fixture_world, load

from requisite.core import FrameItem, Item
from requisite.evidence import extract
from requisite.evidence.items import read_items
from requisite.evidence.regulations_gov import instant


def entry(n: str, posted: str = "2026-05-19T04:00:00Z") -> FrameItem:
    return FrameItem(id=f"{DOCKET}-{n}", docket=DOCKET, posted_at=instant(posted), withdrawn=False)


def test_an_item_is_the_comment_then_each_file_in_order_each_headed_by_its_artifact(live, make_client, store, scan):
    api = fixture_world(FakeAPI(), scanned_0003=scan)
    items = read_items(make_client(api, live / "cassettes", live), store, [entry("0005"), entry("0004"), entry("0003")],
                       workers=3)  # fmt: skip
    united, partners, boeing = items
    assert [i.id for i in items] == [f"{DOCKET}-0005", f"{DOCKET}-0004", f"{DOCKET}-0003"]
    assert (united.submitter, partners.submitter, boeing.submitter) == (
        "United Airlines, Inc", "Aviation Partners Boeing", "Boeing Commercial Airplanes")  # fmt: skip
    assert united.container == DOCKET and united.posted_at == datetime(2026, 5, 19, 4, tzinfo=UTC)
    record, letter = united.artifacts
    letter_read = extract.read("pdf", fixture_pdf())  # the letter as the PDF reader reads it: its method and its tools
    assert united.text.startswith(
        f"[comment {DOCKET}-0005 · {record}]\nDocket No. FAA–2026–2730; Project Identifier AD–2025–01579–T. United "
        f'Airlines (UA) has reviewed and provides comments in attached letter. Thanks.\n\n[attachment 1 "UA Comments '
        f'FAA-2026-01579-T" · attachment_1.pdf · {letter}]\n[page 1 of 1: {letter_read.pages[0]}]\nTechnical Operations\n'
    )
    assert united.extractions == [f"comment/{extract.VERSIONS['comment']}", letter_read.id] and united.unreadable == []
    assert letter_read.id.startswith(f"pdf/{extract.VERSIONS['pdf']} pypdf {pypdf.__version__}; OCR by ")
    assert united.addresses == f"{DOCKET}-0001"  # the document it comments on, as the source states it
    assert store.content(letter) == fixture_pdf()
    assert store.extraction(letter, united.extractions[1]) == united.text.split(f"{letter}]\n", 1)[1]
    assert len(partners.artifacts) == 1 and partners.text.startswith(f"[comment {DOCKET}-0004 · {partners.artifacts[0]}]\nAPB")
    assert "[page 1 of 1: OCR]" in boeing.text and "United Airlines" in boeing.text  # the scan served for 0003
    # Each item is committed, the record published at its postedDate and observed when retrieved.
    kept = store.all("item", Item)
    assert {i.id for i in kept} == {i.id for i in items}
    artifact = store.artifacts_at(f"https://api.regulations.gov/v4/comments/{DOCKET}-0005?include=attachments")[0]
    assert artifact.published_at == united.posted_at and artifact.published_at_basis.startswith("the comment's postedDate")


def attachment(order: int, aid: str, title: str, formats: list | None, **attributes) -> dict:
    a = copy.deepcopy(load("comment_0.json")["body"]["included"][0])
    a["id"] = aid
    a["attributes"].update(docOrder=order, title=title, fileFormats=formats, **attributes)
    return a


def test_files_are_read_in_filing_order_and_what_cannot_be_read_is_stated_on_the_item(live, make_client, store):
    api = fixture_world(FakeAPI())
    body = copy.deepcopy(load("comment_0.json")["body"])
    cid = f"{DOCKET}-0099"
    body["data"]["id"] = cid
    url = "https://downloads.regulations.gov/FAA-2026-2730-0099"
    body["included"] = [  # served out of filing order
        attachment(3, "a3", "Original letter", [{"fileUrl": f"{url}/attachment_3.wpd", "format": "wpd", "size": 9}]),
        attachment(1, "a1", "Letter", [{"fileUrl": f"{url}/attachment_1.docx", "format": "docx", "size": 9},
                                       {"fileUrl": f"{url}/attachment_1.pdf", "format": "pdf", "size": 9}]),
        attachment(2, "a2", "Cost data", None, restrictReasonType="Confidential Business Information"),
        attachment(4, "a4", "Photo", [{"fileUrl": f"{url}/attachment_4.jpg", "format": "jpg", "size": 9}]),
    ]  # fmt: skip
    body["data"]["relationships"]["attachments"]["data"] = [{"id": i, "type": "attachments"} for i in ("a3", "a1", "a2", "a4")]
    api.details[f"comments/{cid}"] = body
    api.files[f"{url}/attachment_1.pdf"] = fixture_pdf()
    api.files[f"{url}/attachment_3.wpd"] = b"WPC\x00 a WordPerfect file"
    # attachment_4.jpg is not served: the file host answers 403
    (item,) = read_items(make_client(api, live / "cassettes", live), store, [entry("0099")])
    order = [part.split("\n", 1)[0] for part in item.text.split("\n\n[")[1:]]
    assert [o.split(" ·")[0].split("]")[0] for o in order] == [
        'attachment 1 "Letter"', 'attachment 2 "Cost data"', 'attachment 3 "Original letter"', 'attachment 4 "Photo"']  # fmt: skip
    assert f"{url}/attachment_1.docx" not in [r.url for r in api.requests]  # the PDF rendition is read where offered
    assert len(item.artifacts) == 3  # the record, the letter's PDF, and the .wpd kept though unread
    assert list(item.unreadable) == [
        'attachment 2 "Cost data": restricted by the agency; Confidential Business Information; no file is published',
        'attachment 3 "Original letter": .wpd: no reader here, and no PDF rendition is offered',
        f'attachment 4 "Photo": the file host answered 403 for {url}/attachment_4.jpg',
    ]
    assert "(its text could not be read: .wpd: no reader here, and no PDF rendition is offered)" in item.text
    # Whose fact each unread part is: the agency's restriction and the host's 403 are the source's; a format no reader
    # here reads is work not done, which stops the item being judged until it is read (core.ItemPart.by).
    assert [(p.label, p.by) for p in item.parts if p.unreadable] == [
        ('attachment 2 "Cost data"', "source"), ('attachment 3 "Original letter"', "reader"),
        ('attachment 4 "Photo"', "source")]  # fmt: skip
    assert item.readable and item.unread_by_us == ['attachment 3 "Original letter": .wpd: no reader here, and no PDF '
                                                   "rendition is offered"]  # fmt: skip
    # Identical bytes filed twice are one artifact; each filing keeps where its own part was retrieved from.
    (wpd,) = [p for p in item.parts if p.by == "reader"]
    assert wpd.locator == f"{url}/attachment_3.wpd"


def test_a_comment_no_longer_served_stays_an_item_from_its_frame_entry(live, make_client, store):
    api = fixture_world(FakeAPI())
    gone = entry("0042", posted="2026-05-11T04:00:00Z")
    (item,) = read_items(make_client(api, live / "cassettes", live), store, [gone])
    assert (item.container, item.posted_at, item.submitter) == (DOCKET, gone.posted_at, None)
    assert len(item.artifacts) == 1 and item.extractions == [] and not item.readable
    assert item.parts[0].by == "source" and item.unreadable[0].startswith("the comment's record: regulations.gov answered 404 for it at ")
    assert store.artifacts_at(f"https://api.regulations.gov/v4/comments/{DOCKET}-0042?include=attachments")[0].published_at is None
