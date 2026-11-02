"""The gold-labelling session (requisite/discovery/gold.py): blind, in the seed's order, unsure kept as unresolved,
resumable, and the drawn items only."""

from datetime import UTC, datetime
from fractions import Fraction as F

import pytest

from requisite import core
from requisite.core.evidence import artifact_id
from requisite.discovery import gold
from requisite.store import Store

T = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
OBJECTIVE = core.WideObjective(
    id="objective", version=1, population="Small businesses under 500 employees.", events="pending cost measures",
    horizon=core.TimeWindow(start=T, end=None, start_closed=True, end_closed=False), loss_definition="higher costs",
    estimand="share of items", unit="item",
)  # fmt: skip
MEASURES = {"D-1": "A proposed rule raising the overtime salary threshold."}


def make_items(store: Store, n: int) -> core.SampleDraw:
    drawn = []
    for i in range(n):
        art = artifact_id(f"comment {i}".encode())
        item = core.Item(id=f"C-{i}", source="regulations.gov", container="D-1", posted_at=T, submitter=f"Firm {i}",
                         artifacts=[art], text=f"Comment text {i}.", extractions=["rg-comment-v1"])  # fmt: skip
        store.commit("item", item.id, item)
        drawn.append(core.DrawnItem(item=item.id, stratum="all.neg", inclusion_prob=F(1, 10)))
    return core.SampleDraw(design=core.Ref(id="design", version=1), phase=2, seed=3, items=drawn)


def session(store, draw, answers):
    shown: list[str] = []
    feed = iter(answers)
    n = gold.label_session(store, draw, OBJECTIVE, MEASURES, labeler="owen", read=lambda _: next(feed),
                           write=shown.append, now=lambda: T)  # fmt: skip
    return n, "\n".join(shown)


def test_a_session_labels_blind_in_the_seeds_order_and_keeps_unsure(tmp_path):
    store = Store(tmp_path / "case", case="case", clock=lambda: T)
    draw = make_items(store, 3)
    # Three items, two questions each: answer, unsure, then stop after the second item's first question.
    n, shown = session(store, draw, ["1", "u", "2", "q"])
    assert n == 3
    labels = store.all("gold_label", core.GoldLabel)
    from collections import Counter

    assert Counter((g.outcome, g.label) for g in labels) == Counter(
        [("labeled", "in_population"), ("unresolved", None), ("labeled", "other_author")]
    )
    assert all(g.blind and g.inclusion_prob == F(1, 10) and g.stratum == "all.neg" for g in labels)
    # The order is the seed's, not the draw's listing order, and the same every time.
    first = shown.split("Item 1 of 3")[1].split("·")[1].strip()
    import random

    order = list(draw.items)
    random.Random(draw.seed).shuffle(order)
    assert first == order[0].item
    # Nothing shown is a model's answer: no probabilities, no judgments exist to show.
    assert "probab" not in shown.lower() and store.all("judgment", core.Judgment) == []
    # Resuming picks up the unanswered question of the item that was stopped, then the rest.
    n2, shown2 = session(store, draw, ["5", "1", "3", "2", "1"])
    assert n2 == 3 and "All drawn items are labelled" in shown2
    assert len(store.all("gold_label", core.GoldLabel)) == 6
    assert gold.pending(store, draw, lambda item: ["voice_in_population", f"exposure_to_measure.{item.container}"]) == []


def test_an_answer_out_of_range_is_asked_again(tmp_path):
    store = Store(tmp_path / "case", case="case", clock=lambda: T)
    draw = make_items(store, 1)
    n, shown = session(store, draw, ["9", "x", "2", "q"])
    assert n == 1 and "Type 1 to 3" in shown


def test_a_drawn_item_that_was_never_read_is_refused(tmp_path):
    store = Store(tmp_path / "case", case="case", clock=lambda: T)
    draw = core.SampleDraw(design=core.Ref(id="design", version=1), phase=2, seed=1,
                           items=[core.DrawnItem(item="missing", stratum="s", inclusion_prob=F(1, 2))])  # fmt: skip
    with pytest.raises(Exception, match="not committed"):
        session(store, draw, ["q"])
