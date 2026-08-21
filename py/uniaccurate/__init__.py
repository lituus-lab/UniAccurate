# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""uniaccurate — Python binding over the UniAccurate C library."""
import numbers
from ._core import two_sum as _two_sum_c, version as _version_c
from ._core import (
    _sum_naive_c,
    _sum_pairwise_c,
    _sum_pairwise_iterative_c,
    _sum_kahan_c,
    _sum_neumaier_c,
    _sum_klein_c,
    _sum_shewchuk_c,
    _sum_exact_c,
    _sum_oro_c,
    _sum_acc_c,
    _sum_near_c,
    _sum_k_c,
    _condition_number_c,
    _dot_exact_c,
    _dot_naive_c,
    _dot2_c,
    _dot_k_c,
    _scaled_norm_c,
    _centered_sum_squares_c,
    _centered_cross_product_c,
)

__version__ = _version_c().decode("ascii")


def two_sum(a, b):
    """Error-free sum: returns (s, e) with a + b == s + e exactly.
    s = fl(a + b) (IEEE-754); e carries the rounding error. Non-finite a or b
    gives s = a + b (IEEE) and e = NaN. Raises TypeError on non-numeric input.
    """
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        raise TypeError(f"a must be a number, got {type(a).__name__}")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        raise TypeError(f"b must be a number, got {type(b).__name__}")
    return _two_sum_c(float(a), float(b))


def naive_sum(values):
    """Naive (sequential) sum. Empty input is 0.0. NaN/Inf propagate.
    Raises TypeError on non-numeric elements. A contiguous float64 buffer
    (array.array('d', ...), a NumPy float64 array, a memoryview, ...) is read
    with no copy; any other iterable (a `list`, most commonly) is validated
    and copied in a single pass.
    """
    return _sum_naive_c(values)


def pairwise_sum(values):
    """Recursive pairwise sum. Empty input is 0.0. NaN/Inf propagate;
    opposite-sign overflow can yield NaN (no fallback). Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_pairwise_c(values)


def pairwise_sum_iterative(values):
    """Iterative (bottom-up) pairwise sum. Empty input is 0.0. NaN/Inf
    propagate. Raises TypeError on non-numeric elements. See `naive_sum` for
    the buffer/list calling convention.
    """
    return _sum_pairwise_iterative_c(values)


def kahan_sum(values):
    """Kahan compensated sum. Empty input is 0.0. NaN/Inf propagate; finite
    inputs never yield NaN (overflow gives ±Inf). Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_kahan_c(values)


def neumaier_sum(values):
    """Kahan-Babuska-Neumaier (magnitude-robust) compensated sum. Empty input
    is 0.0. NaN/Inf propagate; finite inputs never yield NaN. Raises TypeError
    on non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_neumaier_c(values)


def klein_sum(values):
    """Klein two-level compensated sum (~epsilon^2). Empty input is 0.0.
    NaN/Inf propagate; finite inputs never yield NaN. Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_klein_c(values)


def shewchuk_sum(values):
    """Correctly-rounded sum (Shewchuk adaptive-precision expansion, the CPython
    fsum recipe): the exact real sum rounded once under round-to-nearest-even,
    for finite non-overflowing input. Empty input is 0.0. NaN/Inf propagate; an
    intermediate overflow abandons the exact expansion and yields a
    correctly-signed ±Inf (finite inputs never yield NaN). Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_shewchuk_c(values)


def exact_sum(values):
    """Correctly-rounded sum (Neal small superaccumulator: integer exact
    accumulation, single final rounding). Empty input is 0.0. NaN/Inf propagate;
    finite inputs never yield NaN (true overflow gives ±Inf, opposite-sign
    overflow cancels exactly). Raises TypeError on non-numeric elements. See
    `naive_sum` for the buffer/list calling convention.
    """
    return _sum_exact_c(values)


def oro_sum(values):
    """ORO sum2 (Alg 4.1) — magnitude-robust compensated sum, value-identical to
    `neumaier_sum`. Empty input is 0.0. NaN/Inf propagate; finite inputs never
    yield NaN (overflow gives ±Inf). Raises TypeError on non-numeric elements.
    See `naive_sum` for the buffer/list calling convention.
    """
    return _sum_oro_c(values)


def acc_sum(values):
    """Rump AccSum (Alg 4.5) — faithful rounding of the exact sum (within 1 ulp;
    no float lies between the result and the exact sum). Empty input is 0.0.
    NaN/Inf propagate (non-finite input or a sigma0 overflow falls back to the
    exact superaccumulator); finite inputs never yield NaN. Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_acc_c(values)


def near_sum(values):
    """Rump NearSum (Alg 7.4) — correctly-rounded sum: the round-to-nearest of
    the exact real sum, bit-for-bit. Empty input is 0.0. NaN/Inf propagate
    (non-finite input or a sigma0 overflow falls back to the exact
    superaccumulator); finite inputs never yield NaN. Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _sum_near_c(values)


def sum_k(values, k):
    """ORO SumK (Alg 4.8) — K-fold cascaded compensated sum: k=1 naive, k=2
    first-order compensated (≈ `neumaier_sum`), k=3 second-order (≈
    `klein_sum`); `k < 1` is treated as 1. Empty input is 0.0. NaN/Inf propagate;
    finite inputs never yield NaN. Raises TypeError on non-numeric elements and
    `ValueError` if `k` is not an int. See `naive_sum` for the buffer/list
    calling convention.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an int, got {type(k).__name__}")
    return _sum_k_c(values, k)


def condition_number(values):
    """Condition number of the sum, `cond = sum|x_i| / |sum(x_i)|` (Higham 2002,
    §4.1): 1.0 for a no-cancellation sum, large under catastrophic cancellation,
    +inf when the sum is exactly 0 or `sum|x_i|` overflows the float range, 0.0
    for empty input. Finite inputs never yield NaN. Raises TypeError on
    non-numeric elements. See `naive_sum` for the buffer/list calling
    convention.
    """
    return _condition_number_c(values)


def scaled_norm(values):
    """Euclidean norm with scaled squares, preserving representable extremes."""
    return _scaled_norm_c(values)


def centered_sum_squares(values, center):
    """Correctly rounded sum of represented deviations squared."""
    if isinstance(center, bool) or not isinstance(center, numbers.Real):
        raise TypeError(f"center must be a real number, got {type(center).__name__}")
    return _centered_sum_squares_c(values, center)


def centered_cross_product(xs, ys, center_x, center_y):
    """Correctly rounded sum of represented centered pair products."""
    if isinstance(center_x, bool) or not isinstance(center_x, numbers.Real):
        raise TypeError(
            f"center_x must be a real number, got {type(center_x).__name__}")
    if isinstance(center_y, bool) or not isinstance(center_y, numbers.Real):
        raise TypeError(
            f"center_y must be a real number, got {type(center_y).__name__}")
    return _centered_cross_product_c(xs, ys, center_x, center_y)


def exact_dot(xs, ys):
    """Correctly-rounded dot product `sum(x_i * y_i)` (Neal small
    superaccumulator: 64x64->128 product accumulation, single final rounding).
    Empty input is 0.0. NaN/Inf operands propagate; finite operands never yield
    NaN (products held at true magnitude, opposite-sign overflow cancels
    exactly). Raises TypeError on non-numeric elements, ValueError on
    mismatched lengths. See `naive_sum` for the buffer/list calling convention.
    """
    return _dot_exact_c(xs, ys)


def naive_dot(xs, ys):
    """Naive (left-to-right) dot product `sum(x_i * y_i)`. Empty input is 0.0.
    NaN/Inf propagate; finite inputs never yield NaN (a rare opposite-sign
    product overflow to NaN is recovered via the exact superaccumulator). Raises
    TypeError on non-numeric elements, ValueError on mismatched lengths. See
    `naive_sum` for the buffer/list calling convention.
    """
    return _dot_naive_c(xs, ys)


def dot2(xs, ys):
    """Compensated dot product at twice working precision (ORO Alg 5.3, K=2;
    Graillat Dot2FMA) `sum(x_i * y_i)`. Empty input is 0.0. NaN/Inf propagate;
    finite inputs never yield NaN. Raises TypeError on non-numeric elements,
    ValueError on mismatched lengths. See `naive_sum` for the buffer/list
    calling convention.
    """
    return _dot2_c(xs, ys)


def dot_k(xs, ys, k):
    """K-fold compensated dot product (ORO Alg 5.3) `sum(x_i * y_i)`: k=1 the
    naive dot, k=2 twice precision (≈ `dot2`), k=3 threefold; `k < 1` is treated
    as 1. Empty input is 0.0. NaN/Inf propagate; finite inputs never yield NaN.
    Raises TypeError on non-numeric elements, ValueError on mismatched lengths
    or a non-int `k`. See `naive_sum` for the buffer/list calling convention.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an int, got {type(k).__name__}")
    return _dot_k_c(xs, ys, k)


def version():
    """C library version string."""
    return _version_c().decode("ascii")


__all__ = [
    "__version__",
    "acc_sum",
    "condition_number",
    "centered_cross_product",
    "centered_sum_squares",
    "dot2",
    "dot_k",
    "exact_dot",
    "exact_sum",
    "kahan_sum",
    "klein_sum",
    "naive_dot",
    "naive_sum",
    "near_sum",
    "neumaier_sum",
    "oro_sum",
    "pairwise_sum",
    "pairwise_sum_iterative",
    "shewchuk_sum",
    "scaled_norm",
    "sum_k",
    "two_sum",
    "version",
]
