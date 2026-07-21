# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import math

import pytest

import uniaccurate

COMPS = [
    ("kahan_sum", uniaccurate.kahan_sum),
    ("neumaier_sum", uniaccurate.neumaier_sum),
    ("klein_sum", uniaccurate.klein_sum),
    ("shewchuk_sum", uniaccurate.shewchuk_sum),
    ("exact_sum", uniaccurate.exact_sum),
    ("oro_sum", uniaccurate.oro_sum),
    ("acc_sum", uniaccurate.acc_sum),
    ("near_sum", uniaccurate.near_sum),
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
                                     ("klein_sum", uniaccurate.klein_sum),
                                     ("shewchuk_sum", uniaccurate.shewchuk_sum),
                                     ("exact_sum", uniaccurate.exact_sum),
                                     ("oro_sum", uniaccurate.oro_sum),
                                     ("acc_sum", uniaccurate.acc_sum),
                                     ("near_sum", uniaccurate.near_sum)])
def test_magnitude_robust_recovers(name, fn):
    assert fn(MAGNITUDE) == 2.0


@pytest.mark.parametrize("name,fn", COMPS)
def test_finite_overflow_inf_not_nan(name, fn):
    # The Rump family falls back to the exact superaccumulator on overflow;
    # every compensated/correctly-rounded variant gives a single-sign +Inf.
    assert math.isinf(fn([1e308] * 256)) and fn([1e308] * 256) > 0


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


def test_all_compensated_agree_on_exact_data():
    x = list(range(1, 301))
    n = uniaccurate.naive_sum(x)
    assert uniaccurate.kahan_sum(x) == n
    assert uniaccurate.neumaier_sum(x) == n
    assert uniaccurate.klein_sum(x) == n
    assert uniaccurate.shewchuk_sum(x) == n
    assert uniaccurate.exact_sum(x) == n
    assert uniaccurate.oro_sum(x) == n
    assert uniaccurate.acc_sum(x) == n
    assert uniaccurate.near_sum(x) == n


# condition_number is a diagnostic, not a forward-bound sum, so it is checked
# for behavior directly rather than against the oracle bounds above.
def test_condition_number_no_cancellation_is_one():
    assert uniaccurate.condition_number([1.0, 1.0, 1.0, 1.0]) == 1.0


def test_condition_number_zero_sum_is_inf():
    assert math.isinf(uniaccurate.condition_number([1.0, -1.0]))


def test_condition_number_empty_is_zero():
    assert uniaccurate.condition_number([]) == 0.0


def test_condition_number_cancellation_is_large():
    cond = uniaccurate.condition_number([1.0, 1e20, 1.0, -1e20])
    assert cond > 1e19


def test_condition_number_magnitude_overflow_is_inf():
    # sum|x_i| overflows the float range even though the inputs are finite:
    # the exact ratio is not representable, so the diagnostic reports +inf.
    assert math.isinf(uniaccurate.condition_number([1e308] * 256))


def test_condition_number_finite_input_never_nan():
    assert not math.isnan(uniaccurate.condition_number([1e308] * 256))
    assert not math.isnan(uniaccurate.condition_number([1.0, -1.0]))


def test_condition_number_non_numeric_raises():
    with pytest.raises(TypeError):
        uniaccurate.condition_number([1.0, "x", 2.0])
    with pytest.raises(TypeError):
        uniaccurate.condition_number([1.0, None])
    with pytest.raises(TypeError):
        uniaccurate.condition_number([True, 2.0])


# sum_k takes a cascade depth K; K=1 is naive, K>=2 is compensated.
def test_sum_k_recovers_one_on_tenths():
    assert uniaccurate.sum_k([0.1] * 10, 2) == 1.0


def test_sum_k_k1_matches_naive_on_integer_data():
    x = list(range(1, 101))
    assert uniaccurate.sum_k(x, 1) == uniaccurate.naive_sum(x)


def test_sum_k_k0_treated_as_naive():
    # k < 1 is clamped to 1, so k=0 behaves like the naive twoSum chain.
    assert uniaccurate.sum_k([1.0, 2.0, 3.0], 0) == 6.0


def test_sum_k_empty_is_zero():
    assert uniaccurate.sum_k([], 2) == 0.0


def test_sum_k_non_int_k_raises():
    with pytest.raises(ValueError):
        uniaccurate.sum_k([1.0, 2.0], 2.0)
    with pytest.raises(ValueError):
        uniaccurate.sum_k([1.0, 2.0], "2")
    with pytest.raises(ValueError):
        uniaccurate.sum_k([1.0, 2.0], True)


def test_sum_k_non_numeric_raises():
    with pytest.raises(TypeError):
        uniaccurate.sum_k([1.0, "x"], 2)


# Dot product family: naive_dot, dot2, dot_k (k=1 naive, k=2 twice, k=3 threefold).
DOTS = [
    ("naive_dot", uniaccurate.naive_dot),
    ("dot2", uniaccurate.dot2),
]


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_empty_is_zero(name, fn):
    assert fn([], []) == 0.0


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_small_exact(name, fn):
    assert fn([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_int_inputs_promoted(name, fn):
    assert fn([1, 2, 3], [4, 5, 6]) == 32.0


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_integer_valued_exact(name, fn):
    x = list(range(1, 101))
    assert fn(x, x) == sum(i * i for i in x)


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_non_numeric_raises(name, fn):
    with pytest.raises(TypeError):
        fn([1.0, "x"], [2.0, 1.0])
    with pytest.raises(TypeError):
        fn([1.0, None], [2.0, 1.0])


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_mismatched_length_raises(name, fn):
    with pytest.raises(ValueError):
        fn([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_non_finite_propagates(name, fn):
    assert math.isnan(fn([1.0, float("nan")], [2.0, 1.0]))
    assert math.isinf(fn([1.0, float("inf")], [2.0, 1.0]))


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_finite_input_never_nan(name, fn):
    # Products of near-max finite values overflow; opposite-sign overflow would
    # be NaN were it not for the superDot recovery. Finite input ⇒ never NaN.
    m = 1e308
    assert not math.isnan(fn([m, m], [m, -m]))


def test_dot2_recovers_cancellation_lost_by_naive():
    x = [1e20, 1.0, -1e20]
    y = [1.0, 1.0, 1.0]
    assert uniaccurate.dot2(x, y) == 1.0
    assert uniaccurate.naive_dot(x, y) != 1.0


def test_dot2_equals_dot_k2():
    x = [0.1, 0.2, 0.3, 1e20, -1e20]
    y = [0.3, 0.2, 0.1, 1.0, 1.0]
    assert uniaccurate.dot2(x, y) == uniaccurate.dot_k(x, y, 2)


def test_dot_k_k1_matches_naive():
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    assert uniaccurate.dot_k(x, y, 1) == uniaccurate.naive_dot(x, y)


def test_dot_k_k0_treated_as_naive():
    assert uniaccurate.dot_k([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0) == 32.0


def test_dot_k_non_int_k_raises():
    with pytest.raises(ValueError):
        uniaccurate.dot_k([1.0, 2.0], [3.0, 4.0], 2.0)
    with pytest.raises(ValueError):
        uniaccurate.dot_k([1.0, 2.0], [3.0, 4.0], "2")
    with pytest.raises(ValueError):
        uniaccurate.dot_k([1.0, 2.0], [3.0, 4.0], True)


def test_dot_k_non_numeric_raises():
    with pytest.raises(TypeError):
        uniaccurate.dot_k([1.0, "x"], [2.0, 1.0], 2)
