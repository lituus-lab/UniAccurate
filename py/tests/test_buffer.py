# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Zero-copy buffer fast path (array.array('d', ...), memoryview, NumPy
float64 arrays): every function must return the same value as the list
path, and anything not a contiguous float64 buffer must fall back to the
validated list path rather than misreading memory or silently truncating.
"""
import array
import random

import pytest

import uniaccurate

SUMS = [
    ("naive_sum", uniaccurate.naive_sum),
    ("pairwise_sum", uniaccurate.pairwise_sum),
    ("pairwise_sum_iterative", uniaccurate.pairwise_sum_iterative),
    ("kahan_sum", uniaccurate.kahan_sum),
    ("neumaier_sum", uniaccurate.neumaier_sum),
    ("klein_sum", uniaccurate.klein_sum),
    ("shewchuk_sum", uniaccurate.shewchuk_sum),
    ("exact_sum", uniaccurate.exact_sum),
    ("oro_sum", uniaccurate.oro_sum),
    ("acc_sum", uniaccurate.acc_sum),
    ("near_sum", uniaccurate.near_sum),
    ("condition_number", uniaccurate.condition_number),
]

DOTS = [
    ("exact_dot", uniaccurate.exact_dot),
    ("naive_dot", uniaccurate.naive_dot),
    ("dot2", uniaccurate.dot2),
]


def _data(n=1000, seed=7):
    r = random.Random(seed)
    return [r.uniform(-1e6, 1e6) for _ in range(n)]


@pytest.mark.parametrize("name,fn", SUMS)
def test_array_matches_list(name, fn):
    values = _data()
    assert fn(array.array("d", values)) == fn(values)


@pytest.mark.parametrize("name,fn", SUMS)
def test_memoryview_matches_list(name, fn):
    values = _data()
    assert fn(memoryview(array.array("d", values))) == fn(values)


@pytest.mark.parametrize("name,fn", SUMS)
def test_empty_array(name, fn):
    assert fn(array.array("d", [])) == 0.0


def test_sum_k_array_matches_list():
    values = _data()
    buf = array.array("d", values)
    for k in (1, 2, 3):
        assert uniaccurate.sum_k(buf, k) == uniaccurate.sum_k(values, k)


@pytest.mark.parametrize("name,fn", DOTS)
def test_dot_array_matches_list(name, fn):
    xs, ys = _data(seed=1), _data(seed=2)
    assert fn(array.array("d", xs), array.array("d", ys)) == fn(xs, ys)


def test_dot_k_array_matches_list():
    xs, ys = _data(seed=1), _data(seed=2)
    bx, by = array.array("d", xs), array.array("d", ys)
    for k in (1, 2, 3):
        assert uniaccurate.dot_k(bx, by, k) == uniaccurate.dot_k(xs, ys, k)


def test_dot_mismatched_array_lengths_raises():
    with pytest.raises(ValueError):
        uniaccurate.naive_dot(array.array("d", [1.0, 2.0]), array.array("d", [1.0]))


def test_float32_array_falls_back_not_misread():
    # array.array('f', ...) is a genuine 4-byte-per-element buffer: the fast
    # path must reject it (format 'f' != 'd') and fall back to the validated
    # list path, not reinterpret its bytes as float64.
    values = [1.0, 2.0, 3.0]
    f32 = array.array("f", values)
    plain_floats = [float(x) for x in f32]  # what the values are once truncated to f32
    assert uniaccurate.naive_sum(f32) == sum(plain_floats)


def test_non_contiguous_slice_falls_back():
    values = _data(n=2000)
    buf = array.array("d", values)
    mv = memoryview(buf)[::2]  # stride 2 -> non-contiguous
    assert not mv.contiguous
    expected = uniaccurate.naive_sum(values[::2])
    assert uniaccurate.naive_sum(mv) == expected


def test_list_path_unaffected_by_buffer_addition():
    # The original list-based API must still reject non-numeric elements
    # exactly as before -- the buffer fast path is purely additive.
    with pytest.raises(TypeError):
        uniaccurate.naive_sum([1.0, "oops", 2.0])


def test_numpy_float64_matches_list():
    np = pytest.importorskip("numpy")
    values = _data()
    arr = np.array(values, dtype=np.float64)
    assert uniaccurate.shewchuk_sum(arr) == uniaccurate.shewchuk_sum(values)


def test_numpy_float32_falls_back_not_misread():
    np = pytest.importorskip("numpy")
    values = _data(n=100)
    f32 = np.array(values, dtype=np.float32)
    plain_floats = [float(x) for x in f32]
    assert uniaccurate.naive_sum(f32) == sum(plain_floats)


def test_numpy_2d_array_falls_back_and_rejects():
    np = pytest.importorskip("numpy")
    arr2d = np.arange(100, dtype=np.float64).reshape(10, 10)
    with pytest.raises(TypeError):
        uniaccurate.naive_sum(arr2d)


def test_numpy_strided_view_matches_list():
    np = pytest.importorskip("numpy")
    values = _data(n=2000)
    arr = np.array(values, dtype=np.float64)
    strided = arr[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    assert uniaccurate.naive_sum(strided) == uniaccurate.naive_sum(list(strided))
