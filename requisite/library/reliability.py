"""The exact one-sided binomial upper bound on an error rate (addendum A12 §11.4).

Given x errors in n independent gold cases, the exact (Clopper-Pearson) one-sided upper bound at level 1 - alpha is the
error rate p_U at which seeing x or fewer errors has probability alpha:

    P(X <= x | n, p_U) = alpha,  so  p_U = BetaQuantile(1 - alpha; x + 1, n - x)  for x < n, and p_U = 1 for x = n.

With no errors this is 1 - alpha^(1/n): three flawless cases give 1 - 0.05^(1/3) = 0.6316 at 95% (oracle check 13).
That is why three stable instantiations justify scoped reuse and never a reliability claim (A12). The count of
independent cases is the caller's to establish: cases that share sources or templates are not independent, and many
judge configurations on one case are one case (A12 §11.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class ErrorRateBound:
    errors: int
    cases: int
    alpha: float  # one-sided: the bound holds at level 1 - alpha
    upper: float


def error_rate_upper_bound(errors: int, cases: int, *, alpha: float) -> ErrorRateBound:
    """The exact one-sided upper bound at level 1 - alpha on an error rate, from errors in independent cases."""
    for name, v in (("errors", errors), ("cases", cases)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{name} is {v!r}: expected a whole number")
    if not 0 <= errors <= cases:
        raise ValueError(f"{errors} errors in {cases} cases")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha is {alpha!r}: expected a level strictly between 0 and 1")
    # With every case an error (including no cases at all) no rate below 1 is excluded.
    upper = 1.0 if errors == cases else float(beta.ppf(1 - alpha, errors + 1, cases - errors))
    return ErrorRateBound(errors=errors, cases=cases, alpha=alpha, upper=upper)
