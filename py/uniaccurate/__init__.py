# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""uniaccurate — Python binding over the UniAccurate C library."""
from ._core import two_sum as _two_sum_c, version as _version_c
from ._core import (
    _sum_naive_c,
    _sum_pairwise_c,
    _sum_pairwise_iterative_c,
    _sum_kahan_c,
    _sum_neumaier_c,
    _sum_klein_c,
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


def _validate(values):
    """Coerce an iterable of numbers into a list of Python floats.

    Raises TypeError on non-numeric elements (bool is rejected, mirroring
    `two_sum`).
    """
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"elements must be numbers, got {type(v).__name__}")
        out.append(float(v))
    return out


def naive_sum(values):
    """Naive (sequential) sum. Empty input is 0.0. NaN/Inf propagate.
    Raises TypeError on non-numeric elements.
    """
    return _sum_naive_c(_validate(values))


def pairwise_sum(values):
    """Recursive pairwise sum. Empty input is 0.0. NaN/Inf propagate;
    opposite-sign overflow can yield NaN (no fallback). Raises TypeError on
    non-numeric elements.
    """
    return _sum_pairwise_c(_validate(values))


def pairwise_sum_iterative(values):
    """Iterative (bottom-up) pairwise sum. Empty input is 0.0. NaN/Inf
    propagate. Raises TypeError on non-numeric elements.
    """
    return _sum_pairwise_iterative_c(_validate(values))


def kahan_sum(values):
    """Kahan compensated sum. Empty input is 0.0. NaN/Inf propagate; finite
    inputs never yield NaN (overflow gives ±Inf). Raises TypeError on
    non-numeric elements.
    """
    return _sum_kahan_c(_validate(values))


def neumaier_sum(values):
    """Kahan-Babuska-Neumaier (magnitude-robust) compensated sum. Empty input
    is 0.0. NaN/Inf propagate; finite inputs never yield NaN. Raises TypeError
    on non-numeric elements.
    """
    return _sum_neumaier_c(_validate(values))


def klein_sum(values):
    """Klein two-level compensated sum (~epsilon^2). Empty input is 0.0.
    NaN/Inf propagate; finite inputs never yield NaN. Raises TypeError on
    non-numeric elements.
    """
    return _sum_klein_c(_validate(values))


def version():
    """C library version string."""
    return _version_c().decode("ascii")


__all__ = [
    "__version__",
    "kahan_sum",
    "klein_sum",
    "naive_sum",
    "neumaier_sum",
    "pairwise_sum",
    "pairwise_sum_iterative",
    "two_sum",
    "version",
]
