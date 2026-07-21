# cython: language_level=3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
from libc.stdlib cimport malloc, free

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


cdef _sum_raw(list values, double (*fn)(const double *, size_t)):
    """Copy `values` into a malloc'd double buffer, call `fn`, free in finally."""
    cdef Py_ssize_t n = len(values)
    if n == 0:
        return 0.0
    cdef double *buf = <double *>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n):
            buf[i] = values[i]
        return fn(buf, <size_t>n)
    finally:
        free(buf)


def _sum_naive_c(list values):
    """Raw C call: naive sequential sum of `values` (already Python floats)."""
    return _sum_raw(values, ua_sum_naive)


def _sum_pairwise_c(list values):
    """Raw C call: recursive pairwise sum of `values`."""
    return _sum_raw(values, ua_sum_pairwise)


def _sum_pairwise_iterative_c(list values):
    """Raw C call: iterative pairwise sum of `values`."""
    return _sum_raw(values, ua_sum_pairwise_iterative)


def _sum_kahan_c(list values):
    """Raw C call: Kahan compensated sum of `values`."""
    return _sum_raw(values, ua_sum_kahan)


def _sum_neumaier_c(list values):
    """Raw C call: Kahan-Babuska-Neumaier compensated sum of `values`."""
    return _sum_raw(values, ua_sum_neumaier)


def _sum_klein_c(list values):
    """Raw C call: Klein two-level compensated sum of `values`."""
    return _sum_raw(values, ua_sum_klein)


def _sum_shewchuk_c(list values):
    """Raw C call: correctly-rounded (Shewchuk expansion) sum of `values`."""
    return _sum_raw(values, ua_sum_shewchuk)


def _sum_exact_c(list values):
    """Raw C call: correctly-rounded (Neal superaccumulator) sum of `values`."""
    return _sum_raw(values, ua_sum_exact)


def _sum_oro_c(list values):
    """Raw C call: ORO sum2 (magnitude-robust compensated) sum of `values`."""
    return _sum_raw(values, ua_sum_oro)


def _sum_acc_c(list values):
    """Raw C call: Rump AccSum (faithful, within 1 ulp) sum of `values`."""
    return _sum_raw(values, ua_sum_acc)


def _sum_near_c(list values):
    """Raw C call: Rump NearSum (correctly rounded) sum of `values`."""
    return _sum_raw(values, ua_sum_near)


def _condition_number_c(list values):
    """Raw C call: condition number Σ|xᵢ|/|Σxᵢ| of `values`."""
    return _sum_raw(values, ua_condition_number)


cdef _sum_k_raw(list values, int k, double (*fn)(const double *, size_t, int)):
    """Copy `values` into a malloc'd double buffer, call `fn(buf, n, k)`, free."""
    cdef Py_ssize_t n = len(values)
    if n == 0:
        return 0.0
    cdef double *buf = <double *>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n):
            buf[i] = values[i]
        return fn(buf, <size_t>n, k)
    finally:
        free(buf)


def _sum_k_c(list values, int k):
    """Raw C call: ORO SumK (K-fold cascaded compensated) sum of `values`."""
    return _sum_k_raw(values, k, ua_sum_k)


cdef _dot_raw(list xs, list ys, double (*fn)(const double *, const double *, size_t)):
    """Copy `xs`/`ys` into malloc'd double buffers, call `fn`, free in finally."""
    cdef Py_ssize_t n = len(xs)
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
    try:
        for i in range(n):
            bx[i] = xs[i]; by[i] = ys[i]
        return fn(bx, by, <size_t>n)
    finally:
        free(bx); free(by)


def _dot_exact_c(list xs, list ys):
    """Raw C call: correctly-rounded (Neal superaccumulator) dot of `xs`·`ys`."""
    return _dot_raw(xs, ys, ua_dot_exact)


def _dot_naive_c(list xs, list ys):
    """Raw C call: naive (left-to-right) dot of `xs`·`ys`."""
    return _dot_raw(xs, ys, ua_dot_naive)


def _dot2_c(list xs, list ys):
    """Raw C call: twice-precision compensated dot (ORO Alg 5.3, K=2) of `xs`·`ys`."""
    return _dot_raw(xs, ys, ua_dot2)


cdef _dot_k_raw(list xs, list ys, int k,
                double (*fn)(const double *, const double *, size_t, int)):
    """Copy `xs`/`ys` into malloc'd double buffers, call `fn(buf, buf, n, k)`, free."""
    cdef Py_ssize_t n = len(xs)
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
    try:
        for i in range(n):
            bx[i] = xs[i]; by[i] = ys[i]
        return fn(bx, by, <size_t>n, k)
    finally:
        free(bx); free(by)


def _dot_k_c(list xs, list ys, int k):
    """Raw C call: K-fold compensated dot (ORO Alg 5.3) of `xs`·`ys`."""
    return _dot_k_raw(xs, ys, k, ua_dot_k)
