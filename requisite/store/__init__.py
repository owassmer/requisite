"""Committed state: SQLite with a content-addressed blob directory (spec §13 WP1; invariants I1, I2, I4).

Only this package opens the database (tests/store/test_boundary.py holds that). Every write is a commit of a canonical
record and is append-only: a record is never updated or deleted, and writing the same key with other content raises
(spec I2 for evidence; ours for everything else, since committed state that changes in place cannot be recomputed or
audited). What changes is recorded as a new record: a Proposal's status as a new event, an artifact's new content as a
new artifact linked to its predecessor.

Agents never hold a Store. Their only output is a Proposal, which reaches committed state through `submit_proposal`,
where its kind's required gates run (spec I1). Code commits its own records (artifacts, judgments, approvals and the
rest) through the typed `commit_*` methods.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from requisite import core
from requisite.core import (
    AcceptanceRecord,
    Approval,
    Artifact,
    Claim,
    DerivationReceipt,
    GateResult,
    GoldLabel,
    Judgment,
    Override,
    Proposal,
    Record,
    SyntheticArtifact,
    SyntheticClaim,
    sha256,
)
from requisite.core.evidence import artifact_id, synthetic_artifact_id

R = TypeVar("R", bound=Record)


class Conflict(RuntimeError):
    """A write that would change committed state in place: refused (spec I2; append-only)."""


class Refused(ValueError):
    """A write the store's rules do not admit (a synthetic record in a real case, a claim not in its artifact, ...)."""


# A gate checker decides one gate for one proposal kind: it returns the gate's result, never raises for a failed gate.
GateCheck = Callable[["Store", Proposal], GateResult]

# Schema gates: each proposal kind's payload must be a valid record of this type. Kinds whose payload has no core record
# yet are absent, and a proposal of such a kind cannot pass its schema gate (no proposal is admitted unvalidated). A kind
# whose validator is more than its record's registers a schema check, run once the record validates (spec §5: the
# contract layer's validator is the "IR type check, three-valued evaluator"; a contract's IRs are checked by the
# evaluator, which this package does not import).
PAYLOAD: dict[str, type[Record]] = {
    "question": core.AtomicQuestion,
    "proposition": core.Proposition,
    "constraint": core.Constraint,
    "assumption": core.Assumption,
    "claim": core.Claim,
    "decision_spec": core.DecisionSpec,
    "contract": core.ContractFamily,
    "objective": core.WideObjective,
    "entity_model": core.EntityModel,
    "template": core.QuestionTemplate,
    "paraphrase_set": core.ParaphraseSet,
    "frame": core.FrameCriteria,
}

SCHEMA = """
create table if not exists records (
  kind text not null, key text not null, hash text not null, body text not null, committed_at text not null,
  primary key (kind, key)
);
create table if not exists proposal_events (
  proposal text not null, seq integer not null, status text not null, gates text not null, at text not null,
  primary key (proposal, seq)
);
"""


def _with(proposal: Proposal, status: str, gates: list[GateResult]) -> Proposal:
    """The proposal with this status and these gate results, validated as any record is (model_copy would skip the
    check that a status follows from its gates)."""
    return Proposal.model_validate({**proposal.model_dump(), "status": status, "gates": gates})


@dataclass(frozen=True)
class Committed:
    kind: str
    key: str
    hash: str
    committed_at: datetime


def open_case(root: Path, *, clock: Callable[[], datetime] | None = None) -> Store:
    """An existing case's store, opened as the case it was created as: its name and whether it is synthetic are its
    own first record."""
    path = Path(root) / "requisite.db"
    if not path.exists():
        raise Refused(f"no case is committed at {root}")
    db = sqlite3.connect(path)
    try:
        row = db.execute("select body from records where kind = 'case' and key = 'meta'").fetchone()
    finally:
        db.close()
    if row is None:
        raise Refused(f"{root} holds no case record")
    meta = json.loads(row[0])
    return Store(Path(root), case=str(meta["case"]), synthetic=bool(meta["synthetic"]), clock=clock)


class Store:
    """One case's committed state, under `root`: `requisite.db` and `blobs/`. `synthetic` marks a synthetic case (spec
    I4: fuzzing scenarios and simulations); a real case admits no synthetic artifact or claim, and a synthetic case no
    observed one."""

    def __init__(self, root: Path, *, case: str, synthetic: bool = False, clock: Callable[[], datetime] | None = None):
        self.root = Path(root)
        self.case = case
        self.synthetic = synthetic
        self.clock = clock or (lambda: datetime.now(UTC))
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        self._db = sqlite3.connect(self.root / "requisite.db", isolation_level=None)
        self._db.executescript(SCHEMA)
        meta = self._get("case", "meta")
        if meta is None:
            self._put("case", "meta", {"case": case, "synthetic": synthetic})
        elif meta != {"case": case, "synthetic": synthetic}:
            raise Refused(f"{self.root} holds case {meta}, not {{'case': {case!r}, 'synthetic': {synthetic}}}")
        self._gates: dict[tuple[str, str], GateCheck] = {}

    def close(self) -> None:
        self._db.close()

    # ---------------------------------------------------------------- the append-only core

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("begin immediate")
        try:
            yield self._db
        except BaseException:
            self._db.execute("rollback")
            raise
        self._db.execute("commit")

    def _put(self, kind: str, key: str, body: dict[str, Any]) -> Committed:
        text = core.canonical_json(body)
        h = sha256(text)
        now = self.clock()
        with self._tx() as db:
            row = db.execute("select hash, committed_at from records where kind = ? and key = ?", (kind, key)).fetchone()
            if row is not None:
                if row[0] != h:
                    raise Conflict(f"{kind} {key} is committed with other content; committed state is never overwritten")
                return Committed(kind, key, h, datetime.fromisoformat(row[1]))
            db.execute(
                "insert into records (kind, key, hash, body, committed_at) values (?, ?, ?, ?, ?)",
                (kind, key, h, text, now.isoformat()),
            )
        return Committed(kind, key, h, now)

    def _get(self, kind: str, key: str) -> Any | None:
        row = self._db.execute("select body from records where kind = ? and key = ?", (kind, key)).fetchone()
        return None if row is None else json.loads(row[0])

    def _all(self, kind: str) -> list[Any]:
        rows = self._db.execute("select body from records where kind = ? order by committed_at, key", (kind,))
        return [json.loads(r[0]) for r in rows]

    def get(self, kind: str, key: str, cls: type[R]) -> R | None:
        body = self._get(kind, key)
        return None if body is None else cls.model_validate_json(core.canonical_json(body))

    def all(self, kind: str, cls: type[R]) -> list[R]:
        return [cls.model_validate_json(core.canonical_json(b)) for b in self._all(kind)]

    def commit(self, kind: str, key: str, record: Record) -> Committed:
        """Commit a code-produced record under (kind, key). Prefer the typed commit_* methods, which apply the rules
        that kind carries."""
        if kind in {"artifact", "synthetic_artifact", "claim", "synthetic_claim", "proposal", "case", "derivation"}:
            raise Refused(f"{kind} records are committed through their own method, which checks them")
        return self._put(kind, key, record.model_dump(mode="json"))

    # ---------------------------------------------------------------- evidence (spec I2, I4)

    def _blob(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def _write_blob(self, digest: str, content: bytes) -> None:
        path = self._blob(digest)
        if path.exists():
            if path.read_bytes() != content:  # a hash collision, or a blob altered on disk
                raise Conflict(f"blob {digest} on disk is not the content its hash names")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)

    def commit_artifact(self, artifact: Artifact, content: bytes) -> Committed:
        """An observed artifact and its bytes. Its id is the hash of the bytes (I2); a predecessor must be committed."""
        if self.synthetic:
            raise Refused("a synthetic case holds no observed artifact (spec I4)")
        if artifact.id != artifact_id(content) or artifact.size != len(content):
            raise Refused("an artifact's id and size are those of its bytes (spec I2)")
        if artifact.predecessor is not None and self._get("artifact", artifact.predecessor) is None:
            raise Refused(f"predecessor {artifact.predecessor} is not committed")
        self._write_blob(artifact.id.removeprefix("art_"), content)
        return self._put("artifact", artifact.id, artifact.model_dump(mode="json"))

    def artifacts_at(self, locator: str) -> list[Artifact]:
        """The observed artifacts retrieved from this locator, oldest first: a source's versions, each linked to the one
        before (spec I2: "A changed source is a new version linked to its predecessor")."""
        rows = self._db.execute(
            "select body from records where kind = 'artifact' and json_extract(body, '$.locator') = ? "
            "order by committed_at, rowid",
            (locator,),
        )
        return [Artifact.model_validate_json(core.canonical_json(json.loads(r[0]))) for r in rows]

    def content(self, artifact: str) -> bytes:
        digest = artifact.removeprefix("syn_art_").removeprefix("art_")
        return self._blob(digest).read_bytes()

    def commit_extraction(self, artifact: str, extraction: str, text: str) -> Committed:
        """An artifact's text as one versioned extraction produced it (ours: a claim's span is an offset into exactly
        this text)."""
        known = self._get("artifact", artifact) or self._get("synthetic_artifact", artifact)
        if known is None:
            raise Refused(f"artifact {artifact} is not committed")
        return self._put("extraction", f"{artifact}|{extraction}", {"text": text})

    def extraction(self, artifact: str, extraction: str) -> str | None:
        body = self._get("extraction", f"{artifact}|{extraction}")
        return None if body is None else str(body["text"])

    def commit_claim(self, claim: Claim) -> Committed:
        """An observed claim: its text must be its artifact's extracted text at its span (spec §5: Supports, with a
        span)."""
        if self.synthetic:
            raise Refused("a synthetic case holds no observed claim (spec I4)")
        text = self.extraction(claim.artifact, claim.extraction)
        if text is None:
            raise Refused(f"no extraction {claim.extraction} of {claim.artifact} is committed")
        start, end = claim.span
        if text[start:end] != claim.text:
            raise Refused("a claim's text is not its artifact's text at its span")
        return self._put("claim", claim.id, claim.model_dump(mode="json"))

    def commit_synthetic_artifact(self, artifact: SyntheticArtifact, content: bytes) -> Committed:
        if not self.synthetic:
            raise Refused("a real case holds no synthetic artifact (spec I4)")
        if artifact.id != synthetic_artifact_id(content) or artifact.size != len(content):
            raise Refused("a synthetic artifact's id and size are those of its bytes")
        self._write_blob(artifact.id.removeprefix("syn_art_"), content)
        return self._put("synthetic_artifact", artifact.id, artifact.model_dump(mode="json"))

    def commit_synthetic_claim(self, claim: SyntheticClaim) -> Committed:
        if not self.synthetic:
            raise Refused("a real case holds no synthetic claim (spec I4)")
        content = self.content(claim.artifact).decode()
        if content[claim.span[0] : claim.span[1]] != claim.text:
            raise Refused("a synthetic claim's text is not its artifact's text at its span")
        return self._put("synthetic_claim", claim.id, claim.model_dump(mode="json"))

    # ---------------------------------------------------------------- the single write path for agents (spec I1)

    def register_gate(self, kind: str, gate: str, check: GateCheck) -> None:
        """A package registers how it decides a gate for a kind (the tests of a node, the relevance of a question)."""
        if gate not in core.REQUIRED_GATES.get(kind, ()) + core.governance.SPLIT_GATES:
            raise Refused(f"{gate} is not a gate {kind} proposals pass")
        self._gates[(kind, gate)] = check

    def _schema(self, proposal: Proposal) -> GateResult:
        cls = PAYLOAD.get(proposal.kind)
        if cls is None:
            return GateResult(gate="schema", passed=False, detail=f"no record type validates a {proposal.kind} payload")
        try:
            cls.model_validate_json(core.canonical_json(proposal.payload))
        except ValidationError as e:
            return GateResult(gate="schema", passed=False, detail=str(e))
        return GateResult(gate="schema", passed=True, detail=f"a valid {cls.__name__}")

    def _human(self, proposal: Proposal) -> GateResult:
        approvals = [a for a in self.all("approval", Approval) if a.subject.id == proposal.id]
        lineage = sha256(core.canonical_json(proposal.payload))
        current = [a for a in approvals if a.lineage_hash == lineage]
        if not current:
            return GateResult(gate="human", passed=False, detail="awaiting a human approval of this exact payload")
        last = current[-1]
        return GateResult(gate="human", passed=last.decision != "rejected", detail=f"{last.gate} {last.decision}")

    def submit_proposal(self, proposal: Proposal) -> Proposal:
        """The agents' only write path (spec I1). Runs the proposal's required gates in order, stopping at the first that
        fails, and records the result: active when every required gate passed, rejected when an automatic gate failed,
        proposed while it awaits a human. The returned proposal carries its gate results and status."""
        if proposal.case != self.case:
            raise Refused(f"proposal {proposal.id} belongs to case {proposal.case}, not {self.case}")
        if proposal.status != "proposed" or proposal.gates:
            raise Refused("a proposal is submitted as proposed, with no gate results; the gates are the store's to run")
        status, results = self._run_gates(proposal)
        decided = _with(proposal, status, results)
        self._put("proposal", f"{proposal.id}|submitted", proposal.model_dump(mode="json"))
        self._event(proposal.id, status, results)
        return decided

    def _run_gates(self, proposal: Proposal) -> tuple[str, list[GateResult]]:
        """The proposal's required gates in order, stopping at the first that fails: active when every one passed,
        proposed while a human gate awaits a decision, rejected when any other gate failed."""
        results: list[GateResult] = []
        for gate in core.required_gates(proposal.kind, proposal.payload):
            if gate == "schema":
                result = self._schema(proposal)
                if result.passed and (proposal.kind, gate) in self._gates:
                    result = self._gates[(proposal.kind, gate)](self, proposal)
            elif gate == "human":
                result = self._human(proposal)
            elif (proposal.kind, gate) in self._gates:
                result = self._gates[(proposal.kind, gate)](self, proposal)
            else:
                result = GateResult(gate=gate, passed=False, detail=f"no {gate} check is registered for {proposal.kind}")
            results.append(result)
            if not result.passed:
                awaiting = gate == "human" and result.detail.startswith("awaiting")
                return ("proposed" if awaiting else "rejected"), results
        return "active", results

    def _event(self, proposal: str, status: str, gates: list[GateResult]) -> None:
        with self._tx() as db:
            seq = db.execute("select coalesce(max(seq), 0) + 1 from proposal_events where proposal = ?", (proposal,))
            n = seq.fetchone()[0]
            db.execute(
                "insert into proposal_events (proposal, seq, status, gates, at) values (?, ?, ?, ?, ?)",
                (proposal, n, status, json.dumps([g.model_dump(mode="json") for g in gates]), self.clock().isoformat()),
            )

    def proposal(self, proposal_id: str) -> Proposal | None:
        """A proposal as it now stands: its submitted payload with its latest status and gate results."""
        submitted = self._get("proposal", f"{proposal_id}|submitted")
        if submitted is None:
            return None
        row = self._db.execute(
            "select status, gates from proposal_events where proposal = ? order by seq desc limit 1", (proposal_id,)
        ).fetchone()
        gates = [GateResult.model_validate_json(core.canonical_json(g)) for g in json.loads(row[1])]
        return _with(Proposal.model_validate_json(core.canonical_json(submitted)), row[0], gates)

    def resubmit_after_approval(self, proposal_id: str) -> Proposal:
        """Run a proposal's gates again once a human has decided (a G1 approval of a decision_spec)."""
        current = self.proposal(proposal_id)
        if current is None:
            raise Refused(f"no proposal {proposal_id}")
        if current.status != "proposed":
            raise Refused(f"proposal {proposal_id} is {current.status}, not awaiting a human")
        fresh = _with(current, "proposed", [])
        status, results = self._run_gates(fresh)
        self._event(proposal_id, status, results)
        return _with(fresh, status, results)

    # ---------------------------------------------------------------- code's own records

    def commit_judgment(self, judgment: Judgment) -> Committed:
        return self._put("judgment", judgment.content_hash(), judgment.model_dump(mode="json"))

    def commit_approval(self, approval: Approval) -> Committed:
        return self._put("approval", approval.id, approval.model_dump(mode="json"))

    def commit_override(self, override: Override) -> Committed:
        return self._put("override", override.content_hash(), override.model_dump(mode="json"))

    def commit_derivations(self, receipts: Sequence[DerivationReceipt]) -> list[DerivationReceipt]:
        """The receipts of one recompute's computed values, in one transaction (spec §5, I7; addendum A11 §11.2). A
        receipt is keyed by its lineage hash, which names one derivation: a node version applied to inputs of given value
        and lineage. Where a lineage is already committed, the receipt first committed for it stands and is returned in
        place of the new one. A receipt that would give a committed lineage another value or other inputs is refused: a
        pure node's value is fixed by its lineage, so the node changed without a new version (spec §5) or is not pure."""
        standing: list[DerivationReceipt] = []
        now = self.clock().isoformat()
        with self._tx() as db:
            for receipt in receipts:
                row = db.execute(
                    "select body from records where kind = 'derivation' and key = ?", (receipt.lineage_hash,)
                ).fetchone()
                if row is not None:
                    existing = DerivationReceipt.model_validate_json(row[0])
                    if (existing.node, existing.value_hash, existing.inputs) != (
                        receipt.node, receipt.value_hash, receipt.inputs,
                    ):  # fmt: skip
                        raise Conflict(
                            f"lineage {receipt.lineage_hash} was derived by {existing.node.id} v{existing.node.version} "
                            f"to value {existing.value_hash}, now to {receipt.value_hash}: a node changed without a new "
                            "version, or is not pure"
                        )
                    standing.append(existing)
                    continue
                text = core.canonical_json(receipt.model_dump(mode="json"))
                db.execute(
                    "insert into records (kind, key, hash, body, committed_at) values (?, ?, ?, ?, ?)",
                    ("derivation", receipt.lineage_hash, sha256(text), text, now),
                )
                standing.append(receipt)
        return standing

    def commit_gold_label(self, label: GoldLabel) -> Committed:
        key = f"{label.design.id}|{label.design.version}|{label.item}|{label.question.id}|{label.question.version}"
        return self._put("gold_label", key, label.model_dump(mode="json"))

    # An acceptance record is written at decision time (spec §11 step 1): one decided more than this long before it is
    # committed is a reconstruction, refused (ours: the bound allows for the time a commit takes, nothing more).
    ACCEPTANCE_LAG = timedelta(minutes=5)

    def commit_acceptance(self, record: AcceptanceRecord) -> Committed:
        """Immutable and written at decision time (spec §4, §11 step 1). One per case and decision time."""
        if record.case != self.case:
            raise Refused(f"an acceptance record of case {record.case} in case {self.case}")
        lag = self.clock() - record.decided_at
        if lag < timedelta(0) or lag > self.ACCEPTANCE_LAG:
            raise Refused(f"an acceptance record is written when the decision is made, not {lag} after it")
        return self._put("acceptance", record.decided_at.isoformat(), record.model_dump(mode="json"))
