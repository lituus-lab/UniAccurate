# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import math

import pytest

import uniaccurate

COMPS = [
    ("kahan_sum", uniaccurate.kahan_sum),
    ("neumaier_sum", uniaccurate.neumaier_sum),
    ("klein_sum", uniaccurate.klein_sum),
]

# Where Kahan loses a small addend dominated by the running sum, Neumaier and
# Klein recover the exact 2.0.
MAGNITUDE = [1.0, 1e100, 1.0, -1e100]


@pytest.mark.parametrize("name,fn", COMPS)
def test_empty_is_zero(name, fn):
    assert fn([]) == 0.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_small_exact(name, fn):
    assert fn([1.0, 2.0, 3.0, 4.0]) == 10.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_integer_valued_exact(name, fn):
    assert fn(list(range(1, 101))) == 5050.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_int_inputs_promoted(name, fn):
    assert fn([1, 2, 3]) == 6.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_recovers_lost_low_order_bit(name, fn):
    # Naive gives 0.9999999999999999; compensation recovers 1.0.
    assert fn([0.1] * 10) == 1.0


def test_kahan_loses_dominated_addend():
    # Kahan's form loses the small addends — the magnitude-robustness
    # limitation the Neumaier/Klein variants fix.
    assert uniaccurate.kahan_sum(MAGNITUDE) != 2.0


@pytest.mark.parametrize("name,fn", [("neumaier_sum", uniaccurate.neumaier_sum),
                                     ("klein_sum", uniaccurate.klein_sum)])
def test_magnitude_robust_recovers(name, fn):
    assert fn(MAGNITUDE) == 2.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_non_finite_propagates(name, fn):
    assert math.isnan(fn([1.0, float("nan"), 2.0]))
    assert math.isinf(fn([1.0, float("inf")]))


@pytest.mark.parametrize("name,fn", COMPS)
def test_finite_input_never_nan(name, fn):
    # Overflow to a single-sign inf, not NaN. 1e308 x 256 overflows float64.
    assert math.isinf(fn([1e308] * 256))


@pytest.mark.parametrize("name,fn", COMPS)
def test_non_numeric_raises(name, fn):
    with pytest.raises(TypeError):
        fn([1.0, "x", 2.0])
    with pytest.raises(TypeError):
        fn([1.0, None])
    with pytest.raises(TypeError):
        fn([True, 2.0])


def test_all_three_agree_on_exact_data():
    x = list(range(1, 301))
    n = uniaccurate.naive_sum(x)
    assert uniaccurate.kahan_sum(x) == n
    assert uniaccurate.neumaier_sum(x) == n
    assert uniaccurate.klein_sum(x) == n
