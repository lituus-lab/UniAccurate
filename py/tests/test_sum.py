# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import math

import pytest

import uniaccurate

SUMS = [
    ("naive_sum", uniaccurate.naive_sum),
    ("pairwise_sum", uniaccurate.pairwise_sum),
    ("pairwise_sum_iterative", uniaccurate.pairwise_sum_iterative),
]


@pytest.mark.parametrize("name,fn", SUMS)
def test_empty_is_zero(name, fn):
    assert fn([]) == 0.0


@pytest.mark.parametrize("name,fn", SUMS)
def test_small_exact(name, fn):
    assert fn([1.0, 2.0, 3.0, 4.0]) == 10.0


@pytest.mark.parametrize("name,fn", SUMS)
def test_integer_valued_exact(name, fn):
    assert fn(list(range(1, 101))) == 5050.0


@pytest.mark.parametrize("name,fn", SUMS)
def test_int_inputs_promoted(name, fn):
    assert fn([1, 2, 3]) == 6.0


@pytest.mark.parametrize("name,fn", SUMS)
def test_non_finite_propagates(name, fn):
    assert math.isnan(fn([1.0, float("nan"), 2.0]))
    assert math.isinf(fn([1.0, float("inf")]))


def test_pairwise_overflow_is_nan():
    # Two finite blocks overflow to opposite infinities; the merge yields NaN.
    # Locks the deferred-fallback limitation shared with the Nim/C layer.
    x = [1e308] * 128 + [-1e308] * 128
    assert math.isnan(uniaccurate.pairwise_sum(x))


@pytest.mark.parametrize("name,fn", SUMS)
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
    assert uniaccurate.pairwise_sum(x) == n
    assert uniaccurate.pairwise_sum_iterative(x) == n
