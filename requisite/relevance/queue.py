"""Allocating human attention: the review queue (spec §2.4: "Review priority = relevance x (1 - S_arg), with ties broken
by delta-objective"; spec §13 WP7; addendum A01 §2.2).

A question enters the queue for a named decision only where it is relevant to it: its tie-robust regret exceeds the
decision's tolerance (addendum A03 §3.3; an informational question leaves the queue and stays inspectable: spec §2.2,
A02 §3.4). Relevance is that test, one or zero, as spec §2.2's Tier-0 R_flip is, refined by A03 so a solver's tie is no
flip. Instability counts both scores: 1 - min(S_arg, S_prob), so a question whose configurations agree on the top
answer while their probabilities differ materially is not given zero priority because S_arg = 1 (A01 §2.2: "A review
priority based only on 1 - S_arg assigns this probability instability zero weight"). Ties in priority go to the larger
spread of the decision's objective across the question's answers (spec §2.2's delta-objective), an unbounded spread
first; what is still tied is listed by question id, an order that means nothing and is said to.

A relevant question with no instability is in the queue too where its answer is not established (A02 §3.1: judges can
agree that evidence is insufficient, and the question is not thereby resolved for business use; §3.7: application-
blocking unknowns are identified rather than mistaken for agreement): its priority is zero, so it follows every unstable
one, ordered by spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from requisite.core import Ref


def instability(s_arg: float, s_prob: float) -> float:
    """1 - min(S_arg, S_prob): zero only where both scores are one."""
    if not (0 <= s_arg <= 1 and 0 <= s_prob <= 1):
        raise ValueError("stability scores lie in [0, 1]")
    return 1 - min(s_arg, s_prob)


def priority(relevant: bool, s_arg: float, s_prob: float) -> float:
    """Spec §2.4's review priority, relevance x instability, with S_prob counted (A01 §2.2)."""
    return (1.0 if relevant else 0.0) * instability(s_arg, s_prob)


@dataclass(frozen=True)
class Entry:
    """A question in the review queue for one decision: why it is there, its scores, and the spread of the decision's
    objective across its answers (None: unbounded)."""

    question: Ref
    reason: str
    s_arg: float
    s_prob: float
    spread: Fraction | None
    action: str

    @property
    def priority(self) -> float:
        return priority(True, self.s_arg, self.s_prob)


def order(entries: Sequence[Entry]) -> list[Entry]:
    """The queue in review order: priority first, then the spread (unbounded first), then the question id."""

    def key(e: Entry) -> tuple[float, int, Fraction, str, int]:
        unbounded = e.spread is None
        return (-e.priority, 0 if unbounded else 1, -(e.spread or Fraction(0)), e.question.id, e.question.version)

    return sorted(entries, key=key)
