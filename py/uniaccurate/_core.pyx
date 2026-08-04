# cython: language_level=3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import numbers

from libc.stdlib cimport malloc, free


cdef inline bint _is_real_number(object v):
    ## `numbers.Real` (stdlib ABC) accepts NumPy scalar types (float32, int32,
    ## float64, ...) via NumPy's own virtual-subclass registration -- plain
    ## `isinstance(v, (int, float))` rejects them, even though `double[::1]`'s
    ## zero-copy path above already accepts a real NumPy *array* directly; the
    ## per-element path here is what a non-float64-dtype array or a mixed
    ## Python list of NumPy scalars falls through to. `bool` is `Integral`
    ## (hence `Real`) in this ABC hierarchy, so it needs its own exclusion;
    ## `numpy.bool_` is not registered as `Real` at all and needs none.
    return not isinstance(v, bool) and isinstance(v, numbers.Real)


cdef inline Py_ssize_t _checked_len(Py_ssize_t n) except -1:
    ## Reject before `n * sizeof(double)` can overflow `size_t` and wrap to an
    ## undersized `malloc` (CWE-190): only reachable via a custom iterable's
    ## spoofed `__len__` (no real list/array/buffer can hold this many
    ## elements), but `values`'s type is caller-controlled, not fixed here.
    if n < 0 or <size_t>n > (<size_t>-1) // sizeof(double):
        raise MemoryError(f"requested length {n} too large to allocate")
    return n

cdef extern from "UniAccurate.h":
    const char *ua_version()
    void ua_two_sum(double a, double b, double *s, double *e)
    double ua_sum_naive(const double *x, size_t n)
    double ua_sum_pairwise(const double *x, size_t n)
    double ua_sum_pairwise_iterative(const double *x, size_t n)
    double ua_sum_kahan(const double *x, size_t n)
    double ua_sum_neumaier(const double *x, size_t n)
    double ua_sum_klein(const double *x, size_t n)
    double ua_sum_shewchuk(const double *x, size_t n)
    double ua_sum_exact(const double *x, size_t n)
    double ua_dot_exact(const double *x, const double *y, size_t n)
    double ua_dot_naive(const double *x, const double *y, size_t n)
    double ua_dot2(const double *x, const double *y, size_t n)
    double ua_dot_k(const double *x, const double *y, size_t n, int k)
    double ua_sum_oro(const double *x, size_t n)
    double ua_sum_near(const double *x, size_t n)
    double ua_sum_acc(const double *x, size_t n)
    double ua_sum_k(const double *x, size_t n, int k)
    double ua_condition_number(const double *x, size_t n)


def two_sum(double a, double b):
    """Raw C call. Returns (s, e) with a + b == s + e exactly."""
    cdef double s, e
    ua_two_sum(a, b, &s, &e)
    return (s, e)


def version():
    return ua_version()


## Single dispatcher per family, no separate slow/fast implementation: if
## `values` already exposes a contiguous float64 buffer (array.array('d',
## ...), a NumPy float64 array, a memoryview, ...), read it directly -- zero
## copy. Otherwise it's a generic iterable (a `list`, most commonly):
## validate and copy into a malloc'd buffer in the SAME pass, one Cython-
## compiled loop, not a Python-level validate pass followed by a second copy
## pass (measured slower: building the malloc'd buffer straight from a list
## is faster than building an intermediate array.array in Python first,
## since array.array.append() packs each element into the C buffer -- ~2.7x
## the cost of a plain list's pointer-append -- for no benefit when the
## data is about to be copied into yet another buffer anyway).

cdef _sum_generic(values, double (*fn)(const double *, size_t)):
    cdef const double[::1] view
    try:
        view = values
    except (TypeError, ValueError, BufferError):
        pass
    else:
        if view.shape[0] == 0:
            return 0.0
        return fn(&view[0], <size_t>view.shape[0])
    cdef Py_ssize_t n = _checked_len(len(values))
    if n == 0:
        return 0.0
    cdef double *buf = <double *>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i
    cdef object v
    try:
        for i in range(n):
            v = values[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            buf[i] = v
        return fn(buf, <size_t>n)
    finally:
        free(buf)


def _sum_naive_c(values):
    """Naive sequential sum of `values`."""
    return _sum_generic(values, ua_sum_naive)


def _sum_pairwise_c(values):
    """Recursive pairwise sum of `values`."""
    return _sum_generic(values, ua_sum_pairwise)


def _sum_pairwise_iterative_c(values):
    """Iterative pairwise sum of `values`."""
    return _sum_generic(values, ua_sum_pairwise_iterative)


def _sum_kahan_c(values):
    """Kahan compensated sum of `values`."""
    return _sum_generic(values, ua_sum_kahan)


def _sum_neumaier_c(values):
    """Kahan-Babuska-Neumaier compensated sum of `values`."""
    return _sum_generic(values, ua_sum_neumaier)


def _sum_klein_c(values):
    """Klein two-level compensated sum of `values`."""
    return _sum_generic(values, ua_sum_klein)


def _sum_shewchuk_c(values):
    """Correctly-rounded (Shewchuk expansion) sum of `values`."""
    return _sum_generic(values, ua_sum_shewchuk)


def _sum_exact_c(values):
    """Correctly-rounded (Neal superaccumulator) sum of `values`."""
    return _sum_generic(values, ua_sum_exact)


def _sum_oro_c(values):
    """ORO sum2 (magnitude-robust compensated) sum of `values`."""
    return _sum_generic(values, ua_sum_oro)


def _sum_acc_c(values):
    """Rump AccSum (faithful, within 1 ulp) sum of `values`."""
    return _sum_generic(values, ua_sum_acc)


def _sum_near_c(values):
    """Rump NearSum (correctly rounded) sum of `values`."""
    return _sum_generic(values, ua_sum_near)


def _condition_number_c(values):
    """Condition number Σ|xᵢ|/|Σxᵢ| of `values`."""
    return _sum_generic(values, ua_condition_number)


cdef _sum_k_generic(values, int k, double (*fn)(const double *, size_t, int)):
    cdef const double[::1] view
    try:
        view = values
    except (TypeError, ValueError, BufferError):
        pass
    else:
        if view.shape[0] == 0:
            return 0.0
        return fn(&view[0], <size_t>view.shape[0], k)
    cdef Py_ssize_t n = _checked_len(len(values))
    if n == 0:
        return 0.0
    cdef double *buf = <double *>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i
    cdef object v
    try:
        for i in range(n):
            v = values[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            buf[i] = v
        return fn(buf, <size_t>n, k)
    finally:
        free(buf)


def _sum_k_c(values, int k):
    """ORO SumK (K-fold cascaded compensated) sum of `values`."""
    return _sum_k_generic(values, k, ua_sum_k)


cdef _dot_generic(xs, ys, double (*fn)(const double *, const double *, size_t)):
    cdef const double[::1] vx, vy
    try:
        vx = xs
        vy = ys
    except (TypeError, ValueError, BufferError):
        pass
    else:
        if vx.shape[0] != vy.shape[0]:
            raise ValueError("dot product requires equal-length inputs")
        if vx.shape[0] == 0:
            return 0.0
        return fn(&vx[0], &vy[0], <size_t>vx.shape[0])
    cdef Py_ssize_t n = _checked_len(len(xs))
    # Validate the length before the empty early return: an empty `xs` paired
    # with a non-empty `ys` is a mismatch, not the empty dot (which is 0.0).
    if len(ys) != n:
        raise ValueError("dot product requires equal-length inputs")
    if n == 0:
        return 0.0
    cdef double *bx = <double *>malloc(n * sizeof(double))
    cdef double *by = <double *>malloc(n * sizeof(double))
    if bx == NULL or by == NULL:
        free(bx); free(by)
        raise MemoryError()
    cdef Py_ssize_t i
    cdef object v
    try:
        for i in range(n):
            v = xs[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            bx[i] = v
            v = ys[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            by[i] = v
        return fn(bx, by, <size_t>n)
    finally:
        free(bx); free(by)


def _dot_exact_c(xs, ys):
    """Correctly-rounded (Neal superaccumulator) dot of `xs`·`ys`."""
    return _dot_generic(xs, ys, ua_dot_exact)


def _dot_naive_c(xs, ys):
    """Naive (left-to-right) dot of `xs`·`ys`."""
    return _dot_generic(xs, ys, ua_dot_naive)


def _dot2_c(xs, ys):
    """Twice-precision compensated dot (ORO Alg 5.3, K=2) of `xs`·`ys`."""
    return _dot_generic(xs, ys, ua_dot2)


cdef _dot_k_generic(xs, ys, int k,
                     double (*fn)(const double *, const double *, size_t, int)):
    cdef const double[::1] vx, vy
    try:
        vx = xs
        vy = ys
    except (TypeError, ValueError, BufferError):
        pass
    else:
        if vx.shape[0] != vy.shape[0]:
            raise ValueError("dot product requires equal-length inputs")
        if vx.shape[0] == 0:
            return 0.0
        return fn(&vx[0], &vy[0], <size_t>vx.shape[0], k)
    cdef Py_ssize_t n = _checked_len(len(xs))
    if len(ys) != n:
        raise ValueError("dot product requires equal-length inputs")
    if n == 0:
        return 0.0
    cdef double *bx = <double *>malloc(n * sizeof(double))
    cdef double *by = <double *>malloc(n * sizeof(double))
    if bx == NULL or by == NULL:
        free(bx); free(by)
        raise MemoryError()
    cdef Py_ssize_t i
    cdef object v
    try:
        for i in range(n):
            v = xs[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            bx[i] = v
            v = ys[i]
            if not _is_real_number(v):
                raise TypeError(f"elements must be numbers, got {type(v).__name__}")
            by[i] = v
        return fn(bx, by, <size_t>n, k)
    finally:
        free(bx); free(by)


def _dot_k_c(xs, ys, int k):
    """K-fold compensated dot (ORO Alg 5.3) of `xs`·`ys`."""
    return _dot_k_generic(xs, ys, k, ua_dot_k)
