"""Gold labels for a wide run (spec §7.2 step 4, §16: Owen labels a random sample on Monday; addendum A07 §8.1).

The labeller answers the same questions the models answered, in the original wording, for the items the design drew,
in an order the seed fixes, never seeing any model's answer (core.GoldLabel.blind). "Unsure" is an answer of its own,
recorded as unresolved, never dropped or read as zero (A07 §8.1). Labels are committed one at a time, so a session can
stop and resume; nothing chooses which items get labelled but the draw.
"""

from __future__ import annotations

import random
import textwrap
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from requisite import core
from requisite.discovery import labels
from requisite.store import Refused, Store

UNSURE, QUIT = "u", "q"


def pending(store: Store, draw: core.SampleDraw, question_ids: Callable[[core.Item], list[str]]) -> list[core.DrawnItem]:
    """The drawn items with any question still unlabelled, in the seed's order."""
    done = {
        (g.item, g.question.id)
        for g in store.all("gold_label", core.GoldLabel)
        if g.design == draw.design
    }
    order = list(draw.items)
    random.Random(draw.seed).shuffle(order)  # noqa: S311 (an order, not a secret)
    out = []
    for d in order:
        item = store.get("item", d.item, core.Item)
        if item is None:
            raise Refused(f"drawn item {d.item} is not committed")
        if any((d.item, q) not in done for q in question_ids(item)):
            out.append(d)
    return out


def label_session(
    store: Store,
    draw: core.SampleDraw,
    objective: core.WideObjective,
    measures: Mapping[str, str],
    *,
    labeler: str,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Present each pending item's text and its questions; commit each answer as it is given. Returns how many labels
    were committed. `read` and `write` are the terminal's, injectable for tests."""

    def qids(item: core.Item) -> list[str]:
        return [q.id for q in labels.questions(objective, measures, item).values()]

    todo = pending(store, draw, qids)
    write(f"{len(todo)} of {len(draw.items)} drawn items still need labels. You will not see any model's answer.")
    write("For each question, type the number of your answer, 'u' if you are unsure, or 'q' to stop (labels are kept).")
    write("\nPopulation definition (the first question refers to it):\n" + textwrap.fill(objective.population, 110))
    committed = 0
    labelled = {(g.item, g.question.id) for g in store.all("gold_label", core.GoldLabel) if g.design == draw.design}
    for n, drawn in enumerate(todo, 1):
        item = store.get("item", drawn.item, core.Item)
        assert item is not None  # pending() checked
        write("\n" + "=" * 110)
        write(f"Item {n} of {len(todo)}  ·  {item.id}  ·  docket {item.container}  ·  posted {item.posted_at.date()}")
        write(f"Measure: {measures[item.container]}")
        write("-" * 110)
        write(item.text)
        write("-" * 110)
        for q in labels.questions(objective, measures, item).values():
            if (item.id, q.id) in labelled:
                continue
            target = q.target.replace(objective.population, "(the population definition shown at the start)")
            write("\n" + target)
            for i, a in enumerate(q.answers, 1):
                write(f"  {i}. {a.meaning}")
            while True:
                answer = read("> ").strip().lower()
                if answer == QUIT:
                    write(f"Stopped. {committed} labels committed this session.")
                    return committed
                if answer == UNSURE:
                    outcome, label = "unresolved", None
                    break
                if answer.isdigit() and 1 <= int(answer) <= len(q.answers):
                    outcome, label = "labeled", q.answers[int(answer) - 1].value
                    break
                write(f"Type 1 to {len(q.answers)}, '{UNSURE}' or '{QUIT}'.")
            store.commit_gold_label(
                core.GoldLabel(
                    item=item.id,
                    question=q.ref,
                    design=draw.design,
                    stratum=drawn.stratum,
                    inclusion_prob=drawn.inclusion_prob,
                    outcome=outcome,  # type: ignore[arg-type]
                    label=label,
                    labeler=labeler,
                    labeled_at=now(),
                    blind=True,
                )
            )
            committed += 1
    write(f"\nAll drawn items are labelled. {committed} labels committed this session.")
    return committed
