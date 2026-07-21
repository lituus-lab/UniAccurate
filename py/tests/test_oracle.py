# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Forward-error bound checks against the exact rational oracle (oracle.py).

Every assertion is an exact rational comparison: ``S`` and ``E`` come from
``fractions.Fraction`` (no rounding), and the bound is computed in the same
exact arithmetic. A failure here is a real bound violation, not float noise.
"""
import random
from fractions import Fraction
from functools import partial

import pytest

import uniaccurate
from oracle import (
    abs_sum,
    bound_compensated,
    bound_correctly_rounded,
    bound_dot_compensated,
    bound_dot_naive,
    bound_faithful,
    bound_naive,
    bound_pairwise,
    bound_sumk,
    exact_dot,
    exact_sum,
)

# (name, function, bound(n, E, S)). The compensated bound closes over `leading`:
# 3 for Kahan (no final +c), 2 for Neumaier/Klein (final +c applied). The
# correctly-rounded bound uses the exact sum S (0.5 ulp of fl(S)); the faithful
# bound (accSum) is 1 ulp of fl(S); oro_sum is the ORO sum2, value-identical to
# neumaier_sum so the compensated leading=2 bound applies.
ALGS = [
    ("naive_sum", uniaccurate.naive_sum, bound_naive),
    ("pairwise_sum", uniaccurate.pairwise_sum, bound_pairwise),
    ("pairwise_sum_iterative", uniaccurate.pairwise_sum_iterative, bound_pairwise),
    ("kahan_sum", uniaccurate.kahan_sum, partial(bound_compensated, leading=3)),
    ("neumaier_sum", uniaccurate.neumaier_sum, partial(bound_compensated, leading=2)),
    ("klein_sum", uniaccurate.klein_sum, partial(bound_compensated, leading=2)),
    ("shewchuk_sum", uniaccurate.shewchuk_sum, bound_correctly_rounded),
    ("exact_sum", uniaccurate.exact_sum, bound_correctly_rounded),
    ("oro_sum", uniaccurate.oro_sum, partial(bound_compensated, leading=2)),
    ("acc_sum", uniaccurate.acc_sum, bound_faithful),
    ("near_sum", uniaccurate.near_sum, bound_correctly_rounded),
]

SIZES = [1, 2, 3, 5, 10, 50, 100, 500, 1000]

# Fixed PRNG so a failure reproduces.
_RNG = random.Random(20260720)


def _uniform(lo, hi, n):
    return [_RNG.uniform(lo, hi) for _ in range(n)]


def _mixed_magnitude(n):
    # Exponents spread across [-6, 6]: rounds at every step, exercises
    # compensation recovery without overflowing.
    return [_RNG.uniform(-1, 1) * (2.0 ** _RNG.randint(-6, 6)) for _ in range(n)]


def _cancellation(n):
    # Pairs of near-equal opposite-sign values: sum|x_i| >> |S|, so the
    # absolute bound (in E) is the meaningful one.
    out = []
    for _ in range(max(1, n // 2)):
        a = _RNG.uniform(1, 2)
        out.append(a)
        out.append(-a * _RNG.uniform(0.99, 1.01))
    return out


def _raw_inputs():
    cases = [
        ("0.1x10", [0.1] * 10),
        ("magnitude", [1.0, 1e100, 1.0, -1e100]),
        ("range_1_100", [float(i) for i in range(1, 101)]),
    ]
    for n in SIZES:
        cases.append((f"uniform01_{n}", _uniform(0, 1, n)))
        cases.append((f"uniform_pm1_{n}", _uniform(-1, 1, n)))
        cases.append((f"mixed_{n}", _mixed_magnitude(n)))
        cases.append((f"cancel_{n}", _cancellation(n)))
    return cases


# Precompute the exact oracle (S, E) once per case at collection time.
CASES = [(label, xs, exact_sum(xs), abs_sum(xs)) for label, xs in _raw_inputs()]
CASE_IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("name,fn,bound", ALGS, ids=[a[0] for a in ALGS])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_forward_bound(name, fn, bound, case):
    label, xs, s, e = case
    err = abs(Fraction(fn(xs)) - s)
    limit = bound(len(xs), e, s)
    assert err <= limit, (
        f"{name} {label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )


# Correctly-rounded dot product: a single final rounding of the exact real dot,
# so the bound is 0.5 ulp of fl(S) with S = exact_dot(xs, ys).
DOT_CASES = [
    (f"dot_uniform01_{n}", _uniform(0, 1, n), _uniform(0, 1, n)) for n in SIZES
] + [
    (f"dot_mixed_{n}", _mixed_magnitude(n), _mixed_magnitude(n)) for n in SIZES
] + [
    (f"dot_cancel_{n}", _cancellation(n), _uniform(0.5, 1.5, 2 * max(1, n // 2)))
    for n in SIZES
]
DOT_IDS = [c[0] for c in DOT_CASES]


@pytest.mark.parametrize("label,xs,ys", DOT_CASES, ids=DOT_IDS)
def test_dot_exact_forward_bound(label, xs, ys):
    s = exact_dot(xs, ys)
    e = sum(abs(Fraction(x) * Fraction(y)) for x, y in zip(xs, ys, strict=True))
    err = abs(Fraction(uniaccurate.exact_dot(xs, ys)) - s)
    limit = bound_correctly_rounded(len(xs), e, s)
    assert err <= limit, (
        f"{label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )


# Naive and twice-precision compensated dots, checked against the exact dot
# oracle. naive_dot uses the loose magnitude bound (E); dot2 uses the ORO Thm
# 5.4 / Graillat Dot2 bound (2u|S| + 2 gamma_{n+1}(2u)^2 E).
DOT_ALGS = [
    ("naive_dot", uniaccurate.naive_dot, bound_dot_naive),
    ("dot2", uniaccurate.dot2, partial(bound_dot_compensated, K=2)),
]

# Precompute the exact dot oracle (S, E) once per dot case.
DOT_ORACLE = [
    (label, xs, ys, exact_dot(xs, ys),
     sum(abs(Fraction(x) * Fraction(y)) for x, y in zip(xs, ys, strict=True)))
    for label, xs, ys in DOT_CASES
]


@pytest.mark.parametrize("name,fn,bound", DOT_ALGS, ids=[a[0] for a in DOT_ALGS])
@pytest.mark.parametrize("case", DOT_ORACLE, ids=DOT_IDS)
def test_dot_forward_bound(name, fn, bound, case):
    label, xs, ys, s, e = case
    err = abs(Fraction(fn(xs, ys)) - s)
    limit = bound(len(xs), e, s)
    assert err <= limit, (
        f"{name} {label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )


# dot_k takes a cascade depth K; K=1 naive, K=2 twice, K=3 threefold precision.
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("case", DOT_ORACLE, ids=DOT_IDS)
def test_dot_k_forward_bound(k, case):
    label, xs, ys, s, e = case
    err = abs(Fraction(uniaccurate.dot_k(xs, ys, k)) - s)
    limit = bound_dot_compensated(len(xs), e, s, K=k)
    assert err <= limit, (
        f"dot_k K={k} {label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )


# ORO SumK takes a cascade depth K, so it is checked outside the ALGS loop.
# K=1 is the naive twoSum chain (naive bound); K=2 first-order compensated
# (γ²·E); K=3 second-order (γ³·E). Each K is checked against bound_sumk.
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_sum_k_forward_bound(k, case):
    label, xs, s, e = case
    err = abs(Fraction(uniaccurate.sum_k(xs, k)) - s)
    limit = bound_sumk(len(xs), e, s, K=k)
    assert err <= limit, (
        f"sum_k K={k} {label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )
