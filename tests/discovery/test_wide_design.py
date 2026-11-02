"""F4's sample design as decisions.md fixes it ("F4's sample design"): the phase-1 strata, the equal allocation with
whole strata, the gold split, and the docket's measure. Exact rules, so each is checked at its limits."""

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from requisite.discovery.wide import GOLD_SIZE, allocate, band, gold_allocation
from requisite.evidence.frame import measure_document
from requisite.evidence.regulations_gov import Entry


def test_a_docket_falls_in_its_comment_counts_decade():
    assert [band(n) for n in (20, 99, 100, 999, 1_000, 9_999, 10_000, 65_339)] == [
        "under_100", "under_100", "100_to_999", "100_to_999", "1000_to_9999", "1000_to_9999", "10000_plus", "10000_plus",
    ]  # fmt: skip


def test_a_small_stratum_is_taken_whole_and_what_it_leaves_is_shared():
    # a is whole (10 <= 300); then 1,190 over three is 396, so d (300) is whole too; then 890 over two
    assert allocate({"a": 10, "b": 5_000, "c": 5_000, "d": 300}, 1200) == {"a": 10, "b": 445, "c": 445, "d": 300}
    assert allocate({"a": 10, "b": 5_000, "c": 5_000, "d": 400}, 1200) == {"a": 10, "b": 397, "c": 397, "d": 396}
    assert allocate({"a": 10, "b": 5_000, "c": 5_001}, 1200) == {"a": 10, "b": 595, "c": 595}
    assert allocate({"a": 3, "b": 4}, 1200) == {"a": 3, "b": 4}  # a frame this small is a census
    assert allocate({"a": 500, "b": 500, "c": 500}, 1001) == {"a": 334, "b": 334, "c": 333}  # remainder in order
    assert allocate({"a": 28, "b": 29, "c": 29}, 82) == {"a": 28, "b": 27, "c": 27}  # the remainder completes a
    assert allocate({"a": 1, "b": 6, "c": 869, "d": 869}, 1744) == {"a": 1, "b": 6, "c": 869, "d": 868}


@given(st.dictionaries(st.sampled_from("abcd"), st.integers(1, 3_000), min_size=1), st.integers(4, 2_000))
def test_an_allocation_fits_every_stratum_and_spends_the_budget(sizes, total):
    n = allocate(sizes, total)
    assert all(1 <= n[k] <= sizes[k] for k in sizes)
    assert sum(n.values()) == min(total, sum(sizes.values()))
    partial = [n[k] for k in sizes if n[k] < sizes[k]]
    assert not partial or max(partial) - min(partial) <= 1  # every stratum not read whole gets an equal share
    if partial:  # a stratum is read whole only where it is no bigger than the others' share, or than that share and
        # the one item of the indivisible remainder it takes in band order (decisions, "F4's sample design")
        assert all(sizes[k] <= min(partial) + 1 for k in sizes if n[k] == sizes[k])


def test_the_gold_is_split_80_20_in_each_stratum_and_sums_to_its_budget():
    for strata in (["a"], ["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]):
        g = gold_allocation(strata)
        assert sum(g.values()) == GOLD_SIZE and min(g.values()) >= 2
        for h in strata:
            total = g[(h, 0)] + g[(h, 1)]
            assert abs(g[(h, 1)] - total / 5) <= 0.5  # the above-cut fifth, rounded
    assert gold_allocation(["a", "b", "c", "d"]) == {
        ("a", 0): 26, ("a", 1): 7, ("b", 0): 26, ("b", 1): 7, ("c", 0): 26, ("c", 1): 6, ("d", 0): 26, ("d", 1): 6,
    }  # fmt: skip


def doc(id_: str, kind: str, posted: str, end: str | None = None) -> Entry:
    return Entry({"id": id_, "attributes": {"documentType": kind, "postedDate": posted, "commentEndDate": end}}, 0)


def test_a_dockets_measure_is_its_latest_proposed_rule_or_else_its_open_document():
    as_of = datetime(2026, 11, 2, 1, tzinfo=UTC)
    nprm = doc("A-1", "Proposed Rule", "2025-03-01T05:00:00Z")
    snprm = doc("A-2", "Proposed Rule", "2026-01-10T05:00:00Z")
    notice = doc("A-3", "Notice", "2026-10-01T04:00:00Z", end="2026-12-01T04:59:59Z")
    assert measure_document([nprm, notice, snprm], as_of) == snprm
    assert measure_document([notice, doc("A-4", "Notice", "2026-09-01T04:00:00Z", end="2026-10-01T03:59:59Z")], as_of) == notice
    assert measure_document([doc("A-5", "Rule", "2026-05-01T04:00:00Z")], as_of) is None
