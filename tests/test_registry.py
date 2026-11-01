"""The registry and the tests agree: every covered id is a registry item, none is covered twice, no replaced item is
built to (requisite/registry.toml)."""

import pytest

from requisite import registry


def test_registry_and_tests_agree():
    reg, cov = registry.items(), registry.coverage()
    assert registry.problems(reg, cov) == []


def test_never_cut_items_are_tier_zero():
    # spec §13: the never-cut list is what Tier 0 cannot ship without.
    assert all(i.tier == 0 for i in registry.items().values() if i.never_cut)


@pytest.mark.covers("WP0.ci")
def test_ci_runs_the_gates_replaying_only():
    # spec §13 WP0 and §14: CI runs lint, the type check and the tests, never a live provider.
    ci = (registry.ROOT / ".github" / "workflows" / "gates.yml").read_text()
    for step in ("ruff check", "mypy", 'pytest -m "not live"'):
        assert step in ci
    assert 'REQUISITE_REPLAY_ONLY: "1"' in ci
