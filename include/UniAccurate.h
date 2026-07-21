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

/* Compensated sums of n doubles at x. Empty input (n == 0, x may be NULL) is
 * 0. NaN/Inf propagate; finite inputs never yield NaN (overflow ⇒ ±Inf, no
 * Inf - Inf is evaluated). Never raises. Single-threaded, reentrant. Null x
 * with n > 0 is undefined. */
double ua_sum_kahan(const double *x, size_t n);
double ua_sum_neumaier(const double *x, size_t n);
double ua_sum_klein(const double *x, size_t n);

/* Correctly-rounded sum of n doubles at x (Shewchuk adaptive-precision
 * expansion, the CPython fsum recipe): the exact real sum rounded once under
 * round-to-nearest-even, for finite non-overflowing input. Empty input
 * (n == 0, x may be NULL) is 0. An intermediate overflow abandons the exact
 * expansion and yields a correctly-signed ±Inf (finite input never yields NaN);
 * NaN/Inf in the input propagate. Never raises. Single-threaded, reentrant.
 * Null x with n > 0 is undefined. */
double ua_sum_shewchuk(const double *x, size_t n);

/* Correctly-rounded sum of n doubles at x via the Neal small superaccumulator
 * (integer exact accumulation, single final rounding). Empty input is 0.
 * NaN/Inf propagate; finite inputs never yield NaN (true overflow ⇒ ±Inf,
 * opposite-sign overflow cancels exactly). Never raises. Null x with n > 0
 * is undefined. */
double ua_sum_exact(const double *x, size_t n);

/* Correctly-rounded dot product Σ xᵢyᵢ of n pairs via the Neal small
 * superaccumulator (64×64→128 product accumulation, single final rounding).
 * Empty input is 0. NaN/Inf operands propagate; finite operands never yield
 * NaN (products held at true magnitude, opposite-sign overflow cancels
 * exactly). Never raises. Null x or y with n > 0 is undefined. */
double ua_dot_exact(const double *x, const double *y, size_t n);

/* ORO sum2 (Alg 4.1) — the magnitude-robust compensated sum, value-identical to
 * ua_sum_neumaier. Empty input (n == 0, x may be NULL) is 0. NaN/Inf propagate;
 * finite inputs never yield NaN (overflow ⇒ ±Inf). Never raises.
 * Single-threaded, reentrant. Null x with n > 0 is undefined. No SIMD kernel —
 * scalar under -d:simd (ADR-0007). */
double ua_sum_oro(const double *x, size_t n);

/* Rump NearSum (Alg 7.4) — the correctly-rounded sum, the round-to-nearest of
 * the exact real sum bit-for-bit. Empty input (n == 0, x may be NULL) is 0.
 * NaN/Inf propagate (non-finite input or a sigma0 overflow falls back to the
 * exact superaccumulator); finite inputs never yield NaN (overflow ⇒ ±Inf).
 * Never raises. Single-threaded, reentrant. Null x with n > 0 is undefined. No
 * SIMD kernel — scalar under -d:simd (ADR-0007). */
double ua_sum_near(const double *x, size_t n);

/* Rump AccSum (Alg 4.5) — a faithful rounding of the exact sum (within 1 ulp).
 * Empty input (n == 0, x may be NULL) is 0. NaN/Inf propagate (non-finite input
 * or a sigma0 overflow falls back to the exact superaccumulator); finite inputs
 * never yield NaN (overflow ⇒ ±Inf). Never raises. Single-threaded, reentrant.
 * Null x with n > 0 is undefined. No SIMD kernel — scalar under -d:simd
 * (ADR-0007). */
double ua_sum_acc(const double *x, size_t n);

/* ORO SumK (Alg 4.8) — K-fold cascaded compensated sum: K=1 naive, K=2
 * first-order compensated (≈ ua_sum_neumaier), K=3 second-order (≈ ua_sum_klein);
 * k < 1 is treated as 1. Empty input (n == 0, x may be NULL) is 0. NaN/Inf
 * propagate; finite inputs never yield NaN (overflow ⇒ ±Inf). Never raises.
 * Single-threaded, reentrant. Null x with n > 0 is undefined. No SIMD kernel —
 * scalar under -d:simd (ADR-0007). */
double ua_sum_k(const double *x, size_t n, int k);

/* Condition number of the sum, cond = Σ|xᵢ| / |Σxᵢ|: 1 for a no-cancellation
 * sum, large under catastrophic cancellation, +Inf when the sum is exactly 0 or
 * Σ|xᵢ| overflows the float range, 0 for empty input. Finite inputs never yield
 * NaN. Never raises. Single-threaded, reentrant. Null x with n > 0 is
 * undefined. */
double ua_condition_number(const double *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* UNIACCURATE_H */
