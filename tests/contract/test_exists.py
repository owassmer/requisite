"""core.Exists: a criterion over an open class of members, each judged for itself (grammar findings, decisions; the
coordinator's batch, November 2). Kleene over every combination of the members' truths, the list's completeness and
whether the observation cutoff has passed; and what the definition refuses."""

import itertools
from datetime import timedelta

import pytest
from irkit import EXPIRY, contract

from requisite import core as c
from requisite.contract.evaluate import Members, evaluate
from requisite.core.kleene import Kleene

K = (Kleene.TRUE, Kleene.FALSE, Kleene.UNKNOWN)
QUALIFIES = c.Interpretive(
    id="qualifies",
    question_template="condition_holds",
    bindings={"condition": "the instrument is a qualifying action with a qualifying effect"},
)
SOURCES = [
    c.Source(id="register", name="the Federal Register", rank=1, role="official_record"),
    c.Source(id="agency", name="the acting agency's website", rank=2, role="official_record"),
]
EXISTS = c.Exists(
    id="any_action",
    member="an instrument (act, regulation, order, guidance or agreement)",
    domain="the instruments the source agencies document by the expiration",
    sources=SOURCES,
    on_conflict="acting_body_record",
    expr=QUALIFIES,
)
IR = contract(EXISTS)


def expected(members: tuple[Kleene, ...], complete: Kleene, closed: bool) -> Kleene:
    """The rule, written out on its own (core.Exists)."""
    if Kleene.TRUE in members:
        return Kleene.TRUE
    if Kleene.UNKNOWN in members or complete is not Kleene.TRUE or not closed:
        return Kleene.UNKNOWN
    return Kleene.FALSE


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_exists_is_kleene_over_its_members_their_completeness_and_the_cutoff(n):
    for members in itertools.product(K, repeat=n):
        for complete in K:
            for closed in (False, True):
                at = IR.timing.observation_cutoff + (timedelta(seconds=1) if closed else -timedelta(days=1))
                given = {"any_action": Members({f"m{i}": {"qualifies": v} for i, v in enumerate(members)}, complete)}
                ev = evaluate(IR, [], given, at=at)
                assert ev.trigger_truth is expected(members, complete, closed), (members, complete, closed)


def test_a_member_is_given_every_judged_condition_of_the_body_and_nothing_at_top_level():
    with pytest.raises(ValueError, match="give each judged condition"):
        evaluate(IR, [], {"any_action": Members({"m0": {}}, Kleene.TRUE)}, at=EXPIRY)
    with pytest.raises(ValueError, match="not interpretive conditions or readings here"):
        evaluate(IR, [], {"any_action": Members({}, Kleene.TRUE), "qualifies": Kleene.TRUE}, at=EXPIRY)
    with pytest.raises(TypeError, match="its value is its Members"):
        evaluate(IR, [], {"any_action": Kleene.TRUE}, at=EXPIRY)


def test_an_exists_judges_each_member_and_binds_no_member_itself():
    with pytest.raises(ValueError, match="no judged condition"):
        c.Exists(id="x", member="m", domain="d", sources=SOURCES, on_conflict="acting_body_record", expr=c.Cmp(id="cmp", observable="x", op=">", value=c.Number(value=1)))
    bound = QUALIFIES.model_copy(update={"bindings": {**QUALIFIES.bindings, c.MEMBER: "one instrument"}})
    with pytest.raises(ValueError, match="given per member"):
        c.Exists(id="x", member="m", domain="d", sources=SOURCES, on_conflict="acting_body_record", expr=bound)
    with pytest.raises(ValueError, match="names where its members are recorded"):
        c.Exists(id="x", member="m", domain="d", sources=[], on_conflict="acting_body_record", expr=QUALIFIES)
    with pytest.raises(ValueError, match="source ids are distinct"):
        c.Exists(id="x", member="m", domain="d", sources=[SOURCES[0], SOURCES[0]], on_conflict="acting_body_record", expr=QUALIFIES)
    with pytest.raises(ValueError, match="states what a member is"):
        c.Exists(id="x", member=" ", domain="d", sources=SOURCES, on_conflict="acting_body_record", expr=QUALIFIES)


def test_a_fuzz_run_refuses_an_exists_by_name():
    from requisite.fuzz.scenarios import FuzzError, plan

    with pytest.raises(FuzzError, match="ranges over an open class of members"):
        plan(IR)


def test_an_exists_holds_no_exists():
    with pytest.raises(ValueError, match="holds no further Exists"):
        c.Exists(id="outer", member="m", domain="d", sources=SOURCES, on_conflict="acting_body_record", expr=EXISTS)


def test_the_dossier_states_an_exists_and_its_body():
    from requisite.dossier.contracts import IRView

    view = IRView.__new__(IRView)
    said = view.describe(EXISTS)  # HTML: the agency's apostrophe is escaped
    assert "as recorded by the Federal Register, the acting agency" in said and "website, meets this" in said
    assert view.children(EXISTS) == [("for each member: ", QUALIFIES)]
