"""The contract IR's own rules (spec §4; addendum A05 §5.4)."""

import pytest
from core_examples import CONTRACT, RAIN, T0, T1, WET
from pydantic import ValidationError

from requisite import core as c


def test_an_operative_exclusion_names_a_node_the_contract_has():
    fields = CONTRACT.model_dump()
    with pytest.raises(ValidationError, match="do not have|does not have"):
        c.ContractIR(**{**fields, "exclusions": [{"text": "x", "operative": True, "ir_node": "nowhere"}]})
    assert c.ContractIR(**{**fields, "exclusions": [{"text": "x", "operative": True, "ir_node": "wet"}]})


def test_one_id_names_one_node():
    fields = CONTRACT.model_dump()
    clash = c.And(id="both", args=[WET, c.Cmp(id="wet", observable="rain", op="<", value=WET.value)])
    with pytest.raises(ValidationError, match="names two different nodes"):
        c.ContractIR(**{**fields, "trigger": clash.model_dump()})


def test_by_states_its_lower_bound_or_none():
    with pytest.raises(ValidationError, match="states whether it is included"):
        c.By(id="b", expr=WET, since=T0, since_closed=None, deadline=T1, closed=True)
    assert c.By(id="b", expr=WET, since=T0, since_closed=True, deadline=T1, closed=False).since == T0


def test_a_final_vintage_has_a_deadline_or_a_fallback():
    fields = RAIN.model_dump()
    with pytest.raises(ValidationError, match="deadline or a fallback"):
        c.Observable(**{**fields, "revision_policy": "final"})
    assert c.Observable(**{**fields, "revision_policy": "final", "final_fallback": "first_print"})
    assert c.Observable(**{**fields, "revision_policy": "final", "final_deadline": T1})
