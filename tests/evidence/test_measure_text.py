"""What a measure's own text states (requisite/evidence/measure_text.py; core.MeasureText): read whole in consecutive
parts, each statement of who pays what quoted from the part it was read in (checked as text), and either statements or
the reader's account of what the text says instead."""

import pytest
from pydantic import ValidationError

from requisite import core
from requisite.agents.answers import CostStatementsDraft
from requisite.core.evidence import artifact_id
from requisite.discovery import labels
from requisite.evidence.measure_text import parts

COST = "Schools would pay $70,000 for each initial OPT authorization."


def test_parts_are_the_whole_text_in_order_each_within_the_limit_and_ending_at_a_paragraph_where_one_falls():
    paragraphs = [f"Paragraph {i}. " + "word " * (i % 7 + 3) for i in range(200)]
    text = "\n\n".join(paragraphs)
    ps = parts(text, limit=500)
    assert "".join(ps) == text and all(len(p) <= 500 for p in ps) and len(ps) > 1
    assert all(p.endswith("\n\n") for p in ps[:-1])  # each paragraph is read whole: none here is longer than 250
    unbroken = "x" * 1234
    assert parts(unbroken, limit=500) == [unbroken[:500], unbroken[500:1000], unbroken[1000:]]
    assert parts("", limit=500) == []


def test_a_readers_quote_is_checked_as_text_against_the_part_it_read():
    part = f"II. Costs\n\n{COST} The Department estimates no other cost."
    ok = {"quote": COST, "payer": "SEVP-certified schools", "pays_for": "the OPT fee", "direction": "more",
          "amount": "70000", "currency": "USD", "per": "per initial OPT authorization"}  # fmt: skip
    read = CostStatementsDraft.model_validate({"statements": [ok], "none_why": None}, context={"text": part})
    assert read.statements[0].amount == 70000
    with pytest.raises(ValidationError, match="not found"):
        CostStatementsDraft.model_validate({"statements": [{**ok, "quote": COST.replace("$70,000", "$70000")}],
                                            "none_why": None}, context={"text": part})  # fmt: skip
    with pytest.raises(ValidationError, match="not both"):
        CostStatementsDraft.model_validate({"statements": [ok], "none_why": "it amends text"}, context={"text": part})
    with pytest.raises(ValidationError, match="not both"):
        CostStatementsDraft.model_validate({"statements": [], "none_why": None}, context={"text": part})


def test_a_measures_text_states_its_summary_and_who_pays_what_or_why_not_and_the_question_binds_it():
    stated = core.CostStatement(quote=COST, payer="SEVP-certified schools", pays_for="the OPT fee", direction="more",
                                amount="70000", currency="USD", per="per initial OPT authorization")  # fmt: skip
    mt = core.MeasureText(document="ICEB-1", fr_number="2026-20660", summary="This rule proposes new fees for OPT.",
                          statements=[stated], artifacts=[artifact_id(b"r")], readings=["run"])  # fmt: skip
    said = labels.measure_said(mt)
    assert "This rule proposes new fees for OPT." in said and f'"{COST}"' in said and "70000 USD" in said
    with pytest.raises(ValidationError, match="or the reader says"):
        core.MeasureText(document="X", fr_number=None, summary="s", statements=[], artifacts=[artifact_id(b"r")],
                         readings=["run"])  # fmt: skip
    with pytest.raises(ValidationError, match="and not both"):
        core.MeasureText(document="X", fr_number=None, summary=None, statements=[stated], artifacts=[artifact_id(b"r")],
                         readings=["run"])  # fmt: skip
    silent = core.MeasureText(document="X", fr_number=None, summary=None, summary_why="regulations.gov states none",
                              statements=[], none_why="read whole in 1 part(s), it states no cost to anyone",
                              artifacts=[artifact_id(b"r")], readings=["run"])  # fmt: skip
    assert "states no cost to anyone" in labels.measure_said(silent)
    with pytest.raises(ValidationError, match="not empty"):
        silent.model_validate({**silent.model_dump(mode="json"), "none_why": " "})
    for code in ("usd", "U1D", "US"):
        with pytest.raises(ValidationError, match="three capital letters"):
            core.CostStatement(quote=COST, payer="p", pays_for="f", direction="more", amount="1", currency=code)
