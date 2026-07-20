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


def two_sum(double a, double b):
    """Raw C call. Returns (s, e) with a + b == s + e exactly."""
    cdef double s, e
    ua_two_sum(a, b, &s, &e)
    return (s, e)


def version():
    return ua_version()


def _sum_naive_c(list values):
    """Raw C call: naive sequential sum of `values` (already Python floats)."""
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
        return ua_sum_naive(buf, <size_t>n)
    finally:
        free(buf)


def _sum_pairwise_c(list values):
    """Raw C call: recursive pairwise sum of `values`."""
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
        return ua_sum_pairwise(buf, <size_t>n)
    finally:
        free(buf)


def _sum_pairwise_iterative_c(list values):
    """Raw C call: iterative pairwise sum of `values`."""
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
        return ua_sum_pairwise_iterative(buf, <size_t>n)
    finally:
        free(buf)
