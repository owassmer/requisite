"""Terms a filed contract states that the IR now carries as stated (the grammar batch, November 2): an unknown at expiry
left to an exchange rule, a settlement date stated as a deadline, a void transfer only where a void path exists, and a
binary payout paid as stated. Each test's expectation is the rule, from the IR's own fields."""

from datetime import timedelta
from decimal import Decimal as D

import pytest
from irkit import EXPIRY, binary, contract, procedure, timing
from pydantic import ValidationError

from requisite import core as c
from requisite.contract.evaluate import evaluate
from requisite.core.kleene import Kleene

UNREAD = c.Interpretive(id="qualifies", question_template="condition_holds", bindings={"condition": "a condition"})
RULE = "Rule 7.1 of the Exchange's Rulebook"


def test_an_unknown_left_to_an_exchange_rule_stays_pending_on_that_rules_determination():
    ir = contract(UNREAD, resolution=procedure(on_unknown_at_expiry="per_rule", unknown_rule=RULE, void_transfer=None))
    after = EXPIRY + timedelta(days=1)
    ev = evaluate(ir, [], {"qualifies": Kleene.UNKNOWN}, at=after)
    assert (ev.trigger_truth, ev.procedure_status, len(ev.transfers)) == (Kleene.UNKNOWN, "pending", 0)
    assert any(RULE in u and "not among the inputs" in u for u in ev.unresolved)
    # A known trigger is not the rule's to settle.
    assert evaluate(ir, [], {"qualifies": Kleene.TRUE}, at=after).procedure_status == "resolved_yes"
    assert evaluate(ir, [], {"qualifies": Kleene.FALSE}, at=after).procedure_status == "resolved_no"
    with pytest.raises(ValidationError, match="names that rule"):
        procedure(on_unknown_at_expiry="per_rule")
    with pytest.raises(ValidationError, match="names that rule"):
        procedure(unknown_rule=RULE)


def test_a_void_transfer_is_stated_where_the_procedure_can_end_void_and_need_not_be_elsewhere():
    assert procedure(void_transfer=None).void_transfer is None  # unknown at expiry settles NO: no void path
    with pytest.raises(ValidationError, match="states what a void returns"):
        procedure(on_unknown_at_expiry="void", void_transfer=None)
    with pytest.raises(ValidationError, match="states what a void returns"):
        procedure(on_unknown_at_expiry="extend", extension_days=3, after_extension="void", void_transfer=None)


def test_a_settlement_deadline_dates_the_payment_as_a_range_from_resolution_to_the_deadline():
    lagged = timing(settlement_lag_days=1, payment_lag_days=0)
    exact = contract(UNREAD, timing=lagged)
    deadline = contract(UNREAD, timing=lagged.model_copy(update={"settlement_lag": "at_most"}))
    after = EXPIRY + timedelta(days=2)
    (paid,) = evaluate(exact, [], {"qualifies": Kleene.TRUE}, at=after).transfers
    (by,) = evaluate(deadline, [], {"qualifies": Kleene.TRUE}, at=after).transfers
    assert paid.earliest is None and by.at == paid.at  # the latest date is the exact lag's
    assert by.earliest is not None and by.earliest == EXPIRY and by.earliest < by.at  # settled at resolution, at the earliest
    with pytest.raises(ValidationError, match="not after its latest"):
        c.CashTransfer(at=EXPIRY, earliest=EXPIRY + timedelta(seconds=1), amount=D(1), currency="USD", basis="payout")


def test_a_term_left_at_its_default_is_left_out_of_the_record_so_earlier_contracts_hash_as_they_did():
    ir = contract(UNREAD)
    dumped = ir.model_dump(mode="json")
    assert "settlement_lag" not in dumped["timing"] and "unknown_rule" not in dumped["resolution"]
    assert c.ContractIR.model_validate_json(ir.model_dump_json()).content_hash() == ir.content_hash()


def test_a_binary_payout_is_paid_as_stated_and_a_computed_amount_states_its_rounding():
    stated = contract(UNREAD, payout=binary("1.00", rounding_quantum=None, rounding_mode=None))
    (paid,) = evaluate(stated, [], {"qualifies": Kleene.TRUE}, at=EXPIRY + timedelta(days=1)).transfers
    assert paid.amount == D("1.00")
    with pytest.raises(ValidationError, match="states how its amount is rounded"):
        c.Payout(kind="per_unit", unit_value=D(1), quantity=c.ValueAt(id="v", observable="x", at=EXPIRY),
                 currency="USD", payment="at_settlement")  # fmt: skip
    with pytest.raises(ValidationError, match="quantum and its mode together"):
        binary("1.00", rounding_mode=None)
