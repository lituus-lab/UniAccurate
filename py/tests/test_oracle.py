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
from oracle import abs_sum, bound_compensated, bound_naive, bound_pairwise, exact_sum

# (name, function, bound(n, E)). The compensated bound closes over `leading`:
# 3 for Kahan (no final +c), 2 for Neumaier/Klein (final +c applied).
ALGS = [
    ("naive_sum", uniaccurate.naive_sum, bound_naive),
    ("pairwise_sum", uniaccurate.pairwise_sum, bound_pairwise),
    ("pairwise_sum_iterative", uniaccurate.pairwise_sum_iterative, bound_pairwise),
    ("kahan_sum", uniaccurate.kahan_sum, partial(bound_compensated, leading=3)),
    ("neumaier_sum", uniaccurate.neumaier_sum, partial(bound_compensated, leading=2)),
    ("klein_sum", uniaccurate.klein_sum, partial(bound_compensated, leading=2)),
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
    limit = bound(len(xs), e)
    assert err <= limit, (
        f"{name} {label}: |err|={float(err):.6e} > bound={float(limit):.6e} "
        f"(E={float(e):.6e}, n={len(xs)})"
    )
