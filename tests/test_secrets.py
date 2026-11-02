"""Secrets never appear in logs or cassettes (spec §13 WP0, §14), and a replay-only process never gets one."""

import json

import pytest

from requisite import cassettes, secrets

FAKE = "sk-test-0123456789abcdef"


@pytest.fixture
def live_allowed(monkeypatch, tmp_path):
    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setenv("OPENAI_DECISIONS_API_KEY", FAKE)
    secrets._found.cache_clear()
    yield tmp_path
    secrets._found.cache_clear()


@pytest.mark.covers("WP0.secrets")
def test_a_recorded_cassette_holds_no_secret(live_allowed):
    root = live_allowed
    body = {"headers": {"authorization": f"Bearer {FAKE}"}, "question": "q"}

    def live():
        return {"echo": f"key was {FAKE}", "answers": {"q": {"yes": 0.7, "no": 0.3}}}, 0.001

    budget = cassettes.Budget(log=root / "spend.jsonl")
    first = cassettes.call("openai_decisions", "/v1/decisions", body, live, budget=budget, root=root)
    assert not first.replayed
    files = list(root.rglob("*.json")) + [root / "spend.jsonl"]
    for f in files:
        assert FAKE not in f.read_text()
    kept = json.loads(next(root.rglob("*.json")).read_text())
    assert "[REDACTED]" in json.dumps(kept)


def test_replay_returns_the_recorded_response_without_calling(live_allowed, monkeypatch):
    root = live_allowed
    body = {"question": "same"}
    budget = cassettes.Budget(log=root / "spend.jsonl")
    cassettes.call("clef", "/run", body, lambda: ({"p": 0.4}, 0.0), budget=budget, root=root)
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")

    def must_not_run():
        raise AssertionError("a replay called the provider")

    again = cassettes.call("clef", "/run", body, must_not_run, root=root)
    assert again.replayed and again.response == {"p": 0.4}


def test_replay_only_refuses_a_missing_cassette_and_any_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    secrets._found.cache_clear()
    with pytest.raises(cassettes.CassetteMissing):
        cassettes.call("clef", "/run", {"q": "new"}, lambda: ({}, 0.0), root=tmp_path)
    with pytest.raises(secrets.ReplayOnly):
        secrets.secret("CLOUDFLARE_API_TOKEN")


def test_a_live_call_past_the_run_cap_is_refused(live_allowed):
    root = live_allowed
    budget = cassettes.Budget(run_usd=0.005, log=root / "spend.jsonl")
    with pytest.raises(cassettes.OverBudget):
        cassettes.call("clef", "/run", {"q": 1}, lambda: ({}, 0.0), budget=budget, estimate_usd=0.01, root=root)


def test_identical_requests_in_flight_together_get_one_answer_recorded_once(live_allowed):
    # Two threads asking one request at once: the second waits for the first's recording and replays it, so every
    # caller and every later replay reads the same answer (two answers to one request cannot be recorded in one cassette).
    import threading

    started, release = threading.Event(), threading.Event()
    answers = iter(["first", "second"])
    calls = []

    def live():
        calls.append(1)
        started.set()
        release.wait(5)
        return {"answer": next(answers)}, 0.0

    got: list[str] = []

    def ask():
        budget = cassettes.Budget(log=live_allowed / "spend.jsonl")
        got.append(cassettes.call("p", "e", {"q": 1}, live, budget=budget, root=live_allowed).response["answer"])

    a = threading.Thread(target=ask)
    a.start()
    started.wait(5)
    b = threading.Thread(target=ask)
    b.start()
    release.set()
    a.join(5)
    b.join(5)
    assert got == ["first", "first"] and len(calls) == 1
