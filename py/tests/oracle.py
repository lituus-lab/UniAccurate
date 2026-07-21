# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Exact rational oracle for summation forward-error bounds.

The exact real sum ``S`` and the input magnitude ``E = sum|x_i|`` are computed
with ``fractions.Fraction`` (stdlib), so neither the oracle nor the bound is
rounded: the check ``|fl(alg) - S| <= bound(n, E)`` is an exact rational
comparison. This is stronger than an MPFR oracle (high precision, not exact)
and adds no external dependency. It exercises the same Nim algorithms the
library ships, through the C ABI.

Bounds (``u = 2^-53`` binary64 unit roundoff, half the machine epsilon
``2u = 2^-52``; ``E = sum|x_i|``, ``n = len``), first order:

    naive:                (n-1) u / (1 - (n-1) u) * E        [Higham 2002 (4.4)]
    pairwise / iterative: ceil(log2 n) u / (1 - ceil(log2 n) u) * E  [(4.3)]
    kahan (no final +c):  (3u + (4n+6) u^2) * E              [Hallman & Ipsen 2022]
    neumaier/klein (+c):  (2u + (4n+6) u^2) * E              [Higham 2002 (4.8)]

``kahanSum`` returns the running sum without the final ``+ c`` correction, so
its leading constant is ``3u`` (Hallman & Ipsen 2022, Cor. 4.2); ``neumaierSum``
and ``kleinSum`` apply a final correction, so ``2u`` (Higham 2002, (4.8)). The
compensated bounds carry a ``2x`` test safety margin that absorbs the ``O(u^3)``
tail and the ``u^2`` constant uncertainty; they stay far below the naive bound
for ``n >= 6``, so a degenerate uncompensated sum is still caught.

References:
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed., §4.2-4.3. SIAM. ISBN 978-0-89871-521-7.
- Hallman, E., Ipsen, I.C.F. (2022). "Precision-aware Deterministic and
  Probabilistic Error Bounds for Floating Point Summation".
"""
from fractions import Fraction

# float64 unit roundoff (roundTiesToEven): u = 2^-53, half the machine
# epsilon 2u = 2^-52. The published bounds (Higham 2002, Hallman & Ipsen 2022)
# are stated in u, so this is the value the oracle must use.
U = Fraction(1) / (1 << 53)

# Safety margin on the compensated bounds (test only; not part of the theorem).
_COMP_SAFETY = 2


def exact_sum(values):
    """Exact real sum of ``values`` (floats coerced to exact rationals)."""
    return sum(Fraction(v) for v in values)


def abs_sum(values):
    """Exact ``sum|x_i|`` — the magnitude that scales every bound here."""
    return sum(abs(Fraction(v)) for v in values)


def exact_dot(xs, ys):
    """Exact real dot product ``sum(x_i * y_i)`` (floats coerced to rationals).

    ``strict=True`` so unequal-length ``xs`` / ``ys`` raise instead of silently
    truncating to the shorter — a truncation would compute the wrong exact dot
    and hide a test bug.
    """
    return sum(Fraction(x) * Fraction(y) for x, y in zip(xs, ys, strict=True))


def gamma(k: int) -> Fraction:
    """Higham's ``gamma_k = k*u / (1 - k*u)``; ``0`` for ``k <= 0``."""
    if k <= 0:
        return Fraction(0)
    return U * k / (1 - U * k)


def ceil_log2(n: int) -> int:
    """``ceil(log2(n))`` for ``n >= 2``; ``0`` for ``n <= 1`` (one term is exact)."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def bound_naive(n: int, e: Fraction, s: Fraction = 0) -> Fraction:
    """Recursive (naive) summation forward bound: ``gamma_{n-1} * E``.

    ``s`` (the exact sum) is unused — the naive bound is in the magnitude ``E``.
    """
    return gamma(n - 1) * e


def bound_pairwise(n: int, e: Fraction, s: Fraction = 0) -> Fraction:
    """Pairwise (cascade) forward bound — rounding depth ``ceil(log2 n)``.

    ``s`` is unused (see ``bound_naive``).
    """
    return gamma(ceil_log2(n)) * e


def bound_compensated(n: int, e: Fraction, s: Fraction = 0, *,
                      leading: int) -> Fraction:
    """Compensated summation forward bound.

    ``leading`` is ``3`` for Kahan (no final correction, Hallman & Ipsen 2022)
    and ``2`` for Neumaier/Klein (final ``+ c`` applied, Higham 2002 (4.8)).
    ``s`` is unused (see ``bound_naive``).
    """
    return _COMP_SAFETY * (U * leading + (4 * n + 6) * U * U) * e


def _pow2(e: int) -> Fraction:
    """Exact ``2**e`` as a ``Fraction`` (``e`` may be negative)."""
    if e >= 0:
        return Fraction(1 << e)
    return Fraction(1, 1 << -e)


def _floor_log2(x: Fraction) -> int:
    """Largest ``e`` with ``2**e <= x`` for positive ``x``."""
    num, den = x.numerator, x.denominator
    # 2**e <= num/den  <=>  2**e * den <= num. Compare in integers, scaling both
    # sides by 2**k = 2**max(0, -e) so the left shift 1 << (e+k) stays non-
    # negative even when e is decremented into a dyadic gap (e.g. x = 1/3): k is
    # recomputed every iteration, never fixed once at the entry estimate.
    e = num.bit_length() - den.bit_length()
    while True:
        k = max(0, -e)
        lhs = (1 << (e + k)) * den  # = 2**(e+k) * den
        rhs = num << k              # = num * 2**k
        if lhs <= rhs:
            # e fits; check whether e+1 also fits before accepting.
            k1 = max(0, -(e + 1))
            if (1 << (e + 1 + k1)) * den <= (num << k1):
                e += 1
                continue
            return e
        e -= 1


def bound_correctly_rounded(n: int, e: Fraction, s: Fraction = 0) -> Fraction:
    """Correctly-rounded forward bound: ``0.5 * ulp(fl(S))``.

    A single round-to-nearest of the exact sum ``S`` errs by at most half a unit
    in the last place of the rounded result. ``ulp(fl(S)) <= 2**(e-51)`` for
    ``2**e <= |S| < 2**(e+1)`` (the rounded result may cross the bin boundary
    upward), so ``0.5 * ulp(fl(S)) <= 2**(e-52) = (2u) * 2**e`` — a rigorous,
    at most 2x loose upper bound, computed in exact arithmetic. Subnormal ``S``
    gets the half subnormal-ulp ``2**-1074``; ``S == 0`` is exact (bound ``0``).
    """
    if s == 0:
        return Fraction(0)
    e = _floor_log2(abs(s))
    if e < -1022:  # subnormal range: ulp is the smallest subnormal
        return Fraction(1, 1 << 1074)
    return (2 * U) * _pow2(e)


def bound_faithful(n: int, e: Fraction, s: Fraction = 0) -> Fraction:
    """Faithful forward bound: ``1 * ulp(fl(S))`` — no float lies between the
    result and the exact sum, so the error is under one unit in the last place.
    Twice ``bound_correctly_rounded`` (which already bounds ``0.5 ulp`` at up to
    2x loose), so this bounds ``1 ulp`` at up to 2x loose — rigorous in exact
    arithmetic. ``s`` is the exact sum (as for ``bound_correctly_rounded``).
    """
    return 2 * bound_correctly_rounded(n, e, s)


def bound_sumk(n: int, e: Fraction, s: Fraction = 0, *, K: int) -> Fraction:
    """ORO ``SumK`` forward bound (ORO Thm 4.9): the real-arithmetic K-fold
    cascade error ``gamma_{n-1}^K * E / (1 - gamma_{n-1}^K)`` (so the error is
    ``(n*u)^K * E``), **plus** a final-rounding budget of ``1 ulp of |S|``.

    The cascade term is the documented bound on the *unrounded* distillation; for
    ``K >= 2`` it falls below the single rounding the float result incurs (and
    that the oracle incurs rounding the real sum), neither of which the
    real-arithmetic bound includes. The budget ``2u * |S|`` covers both, so the
    check is the documented bound rather than a loose margin: ``K=1`` is the
    naive bound (cascade term dominates), ``K >= 2`` is K-fold precision (budget
    dominates).
    """
    g = gamma(n - 1)
    gk = g ** K
    return gk * e / (1 - gk) + 2 * U * abs(s)


def gamma_2u(k: int) -> Fraction:
    """Higham's ``gamma_k`` at the doubled unit roundoff ``2u`` (the FMA product
    error in the dot recurrence is bounded at ``2u``): ``(2u)*k / (1 - (2u)*k)``;
    ``0`` for ``k <= 0``.
    """
    if k <= 0:
        return Fraction(0)
    return (2 * U) * k / (1 - (2 * U) * k)


def bound_dot_naive(n: int, e: Fraction, s: Fraction = 0) -> Fraction:
    """Naive dot forward bound. Each product ``fl(x_i y_i)`` errs by at most
    ``u * |x_i y_i|`` (one rounding), and the ``n`` products are then summed
    naively (a further ``gamma_{n-1}`` on the rounded products) — at most
    ``2n`` roundings in total, so ``gamma_{2n} * E`` bounds the absolute error
    in the magnitude ``E = sum|x_i y_i|``. ``s`` is unused (the bound is in
    ``E``); the bound is loose on purpose (the naive dot can be very inaccurate
    under cancellation, where ``E >> |S|``).
    """
    return gamma(2 * n) * e


def bound_dot_compensated(n: int, e: Fraction, s: Fraction, *,
                          K: int) -> Fraction:
    """Compensated dot forward bound (ORO Thm 5.4 / Graillat Dot2):
    ``2u * |S| + 2 * gamma_{n+1}(2u)^K * E`` — a final rounding of ``|S|`` plus
    the K-fold cascade error in the magnitude ``E = sum|x_i y_i|``. ``K = 2`` is
    twice working precision (Graillat Dot2); larger ``K`` gives K-fold. The
    ``2u * |S|`` term is the unavoidable final rounding; the cascade term is
    ``(n * 2u)^K * E``, far below it for ``K >= 2`` at these ``n``.
    """
    return 2 * U * abs(s) + 2 * gamma_2u(n + 1) ** K * e
