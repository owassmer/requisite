"""Every canonical record round-trips through JSON unchanged, with a stable content hash (spec §13 WP1)."""

import inspect

import pytest
from core_examples import examples

from requisite import core
from requisite.core import base, contract, evidence, governance, judgment, results, world


def record_classes():
    modules = (base, contract, evidence, governance, judgment, results, world)
    return {
        cls
        for m in modules
        for _, cls in inspect.getmembers(m, inspect.isclass)
        if issubclass(cls, core.Record) and cls is not core.Record and cls.__module__ == m.__name__
    }


@pytest.mark.covers("WP1.roundtrip")
def test_every_record_round_trips_and_hashes_the_same():
    instances = examples()
    # Every record type the definition declares has an instance here, so none goes unexercised.
    assert record_classes() == {type(x) for x in instances}
    for x in instances:
        again = type(x).model_validate_json(x.model_dump_json())
        assert again == x, type(x).__name__
        assert again.content_hash() == x.content_hash(), type(x).__name__


def test_records_are_immutable_all_the_way_down():
    # spec §4: an AcceptanceRecord is immutable; nothing in it can be edited in place.
    record = next(x for x in examples() if isinstance(x, core.AcceptanceRecord))
    with pytest.raises(TypeError, match="immutable"):
        record.scenario_probabilities["s3"] = 0  # type: ignore[index]
    with pytest.raises(AttributeError):
        record.approvals.append("a2")  # type: ignore[attr-defined]
    proposal = next(x for x in examples() if isinstance(x, core.Proposal))
    with pytest.raises(TypeError, match="immutable"):
        proposal.payload["id"] = "other"  # type: ignore[index]


def test_records_are_frozen_and_closed():
    q = examples()[0]
    with pytest.raises(Exception, match="frozen"):
        q.id = "other"  # type: ignore[misc]
    with pytest.raises(Exception, match="Extra inputs"):
        core.Ref(id="x", version=1, note="a field no package declared")  # type: ignore[call-arg]
