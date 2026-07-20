// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#ifndef UNIACCURATE_H
#define UNIACCURATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UNIACCURATE_VERSION_MAJOR 0
#define UNIACCURATE_VERSION_MINOR 1
#define UNIACCURATE_VERSION_PATCH 0
#define UNIACCURATE_VERSION "0.1.0"

#define UNIACCURATE_VERSION_AT_LEAST(ma, mi, pa) \
  ((UNIACCURATE_VERSION_MAJOR > (ma)) || \
   (UNIACCURATE_VERSION_MAJOR == (ma) && UNIACCURATE_VERSION_MINOR > (mi)) || \
   (UNIACCURATE_VERSION_MAJOR == (ma) && UNIACCURATE_VERSION_MINOR == (mi) && \
    UNIACCURATE_VERSION_PATCH >= (pa)))

/* Static version string; do not free. */
const char *ua_version(void);

/* Error-free sum: writes *s = fl(a+b) and *e with a + b = *s + *e exactly
 * (real arithmetic). For non-finite a or b, *s = a + b (IEEE) and *e = NaN.
 * Never raises. Single-threaded, reentrant. Null s or e is undefined. */
void ua_two_sum(double a, double b, double *s, double *e);

/* Sequential / pairwise sums of n doubles at x. Empty input (n == 0, x may be
 * NULL) is 0. NaN/Inf propagate; opposite-sign overflow can yield NaN (no
 * fallback). Never raises. Single-threaded, reentrant. Null x with n > 0 is
 * undefined. */
double ua_sum_naive(const double *x, size_t n);
double ua_sum_pairwise(const double *x, size_t n);
double ua_sum_pairwise_iterative(const double *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* UNIACCURATE_H */
