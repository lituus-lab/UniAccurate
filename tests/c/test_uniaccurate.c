// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stddef.h>
#include "UniAccurate.h"

static int failures = 0;

static void check_str(const char *name, const char *got, const char *want) {
  if (strcmp(got, want) != 0) { printf("FAIL %s: got \"%s\" want \"%s\"\n", name, got, want); failures++; }
  else printf("ok   %s = \"%s\"\n", name, got);
}

static void two_sum(const char *name, double a, double b, double ws, double we) {
  double s, e;
  ua_two_sum(a, b, &s, &e);
  if (s != ws || e != we) { printf("FAIL %s: got (%g, %g) want (%g, %g)\n", name, s, e, ws, we); failures++; }
  else printf("ok   %s = (%g, %g)\n", name, s, e);
}

static void sum_call(const char *name, double (*f)(const double *, size_t),
                     const double *x, size_t n, double want) {
  double got = f(x, n);
  if (got != want) { printf("FAIL %s: got %g want %g\n", name, got, want); failures++; }
  else printf("ok   %s = %g\n", name, got);
}

static void dot_call(const char *name, double (*f)(const double *, const double *, size_t),
                     const double *x, const double *y, size_t n, double want) {
  double got = f(x, y, n);
  if (got != want) { printf("FAIL %s: got %g want %g\n", name, got, want); failures++; }
  else printf("ok   %s = %g\n", name, got);
}

static void sum_k_call(const char *name, double (*f)(const double *, size_t, int),
                       const double *x, size_t n, int k, double want) {
  double got = f(x, n, k);
  if (got != want) { printf("FAIL %s: got %g want %g\n", name, got, want); failures++; }
  else printf("ok   %s = %g\n", name, got);
}

static void dot_k_call(const char *name,
                       double (*f)(const double *, const double *, size_t, int),
                       const double *x, const double *y, size_t n, int k,
                       double want) {
  double got = f(x, y, n, k);
  if (got != want) { printf("FAIL %s: got %g want %g\n", name, got, want); failures++; }
  else printf("ok   %s = %g\n", name, got);
}

/* Condition number: +Inf is a valid result (zero sum or magnitude overflow), so
 * compare by class as well as value. */
static void cond_call(const char *name, double (*f)(const double *, size_t),
                      const double *x, size_t n, double want) {
  double got = f(x, n);
  int ok = (isnan(got) && isnan(want))
        || (isinf(got) && isinf(want) && (got > 0) == (want > 0))
        || (got == want);
  if (!ok) { printf("FAIL %s: got %g want %g\n", name, got, want); failures++; }
  else printf("ok   %s = %g\n", name, got);
}

int main(void) {
  two_sum("1 + 2", 1.0, 2.0, 3.0, 0.0);
  two_sum("1 + 2e16", 1.0, 2e16, 2e16, 1.0);
  { double s, e;
    ua_two_sum(INFINITY, 1.0, &s, &e);
    if (isinf(s) && isnan(e)) printf("ok   inf + 1 = (inf, nan)\n");
    else { printf("FAIL inf + 1: got (%g, %g) want (inf, nan)\n", s, e); failures++; } }
  check_str("version", ua_version(), UNIACCURATE_VERSION);

  /* Sums: empty -> 0; integer-valued data is exact for all three variants. */
  double small[] = {1.0, 2.0, 3.0, 4.0};
  sum_call("naive [1..4]", ua_sum_naive, small, 4, 10.0);
  sum_call("pairwise [1..4]", ua_sum_pairwise, small, 4, 10.0);
  sum_call("pairwise_iter [1..4]", ua_sum_pairwise_iterative, small, 4, 10.0);
  sum_call("naive empty", ua_sum_naive, NULL, 0, 0.0);
  sum_call("pairwise empty", ua_sum_pairwise, NULL, 0, 0.0);
  sum_call("pairwise_iter empty", ua_sum_pairwise_iterative, NULL, 0, 0.0);

  /* Compensated sums: integer-valued data is exact for all three variants. */
  sum_call("kahan [1..4]", ua_sum_kahan, small, 4, 10.0);
  sum_call("neumaier [1..4]", ua_sum_neumaier, small, 4, 10.0);
  sum_call("klein [1..4]", ua_sum_klein, small, 4, 10.0);
  sum_call("kahan empty", ua_sum_kahan, NULL, 0, 0.0);
  sum_call("neumaier empty", ua_sum_neumaier, NULL, 0, 0.0);
  sum_call("klein empty", ua_sum_klein, NULL, 0, 0.0);

  /* Correctly-rounded: integer-valued data is exact, empty is 0. */
  sum_call("shewchuk [1..4]", ua_sum_shewchuk, small, 4, 10.0);
  sum_call("shewchuk empty", ua_sum_shewchuk, NULL, 0, 0.0);

  /* Correctly-rounded wins the 0.1·10 case (naive yields 0.999...). */
  double tenths[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  sum_call("shewchuk 0.1x10", ua_sum_shewchuk, tenths, 10, 1.0);

  /* Correctly-rounded recovers the sum under catastrophic cancellation. */
  double cancel[] = {1.0, 1e20, 1.0, -1e20};
  sum_call("shewchuk cancel", ua_sum_shewchuk, cancel, 4, 2.0);

  /* Neal superaccumulator: same correctly-rounded cases as shewchuk, plus
   * huge-magnitude exact cancellation (max + max - max = max; naive hits
   * +Inf + -Inf = NaN) and true finite overflow to +Inf. */
  sum_call("exact [1..4]", ua_sum_exact, small, 4, 10.0);
  sum_call("exact empty", ua_sum_exact, NULL, 0, 0.0);
  sum_call("exact 0.1x10", ua_sum_exact, tenths, 10, 1.0);
  sum_call("exact cancel", ua_sum_exact, cancel, 4, 2.0);
  double m = DBL_MAX;
  double hugeCancel[] = {m, m, -m};
  sum_call("exact max+max-max", ua_sum_exact, hugeCancel, 3, m);
  { double g = ua_sum_exact(hugeCancel, 2); /* m + m: true overflow */
    if (isinf(g) && g > 0) printf("ok   exact max+max = +inf\n");
    else { printf("FAIL exact max+max: got %g want +inf\n", g); failures++; } }

  /* Exact dot: integer-exact, empty, product-overflow held at true magnitude
   * (1e100·1e100 = 1e200 representable, minus the small term recovered), and
   * opposite-sign finite overflow that cancels exactly (never NaN). */
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 5.0, 6.0};
  dot_call("exact dot [1..3]·[4..6]", ua_dot_exact, a, b, 3, 32.0);
  dot_call("exact dot empty", ua_dot_exact, NULL, NULL, 0, 0.0);
  double pb[] = {1e100, 1.0};
  double pc[] = {1e100, -1.0};
  dot_call("exact dot product-overflow held", ua_dot_exact, pb, pc, 2, 1e100 * 1e100 - 1.0);
  double px[] = {m, -m};
  double py[] = {m, m};
  dot_call("exact dot opposite-sign cancel", ua_dot_exact, px, py, 2, 0.0);

  /* Naive / dot2 / dotK: integer-exact, empty 0, dot2 recovers the cancellation
   * naive loses (the dot analogue of 0.1·10), dot2 == dotK K=2 bit-for-bit,
   * dotK K=1 == naive, K<1 treated as 1, and finite opposite-sign product
   * overflow never NaN (recovered via superDot through the naiveDot fallback). */
  dot_call("naive dot [1..3]·[4..6]", ua_dot_naive, a, b, 3, 32.0);
  dot_call("naive dot empty", ua_dot_naive, NULL, NULL, 0, 0.0);
  dot_call("dot2 [1..3]·[4..6]", ua_dot2, a, b, 3, 32.0);
  dot_call("dot2 empty", ua_dot2, NULL, NULL, 0, 0.0);
  double dcx[] = {1e20, 1.0, -1e20};
  double dcy[] = {1.0, 1.0, 1.0};
  dot_call("dot2 recovers cancel", ua_dot2, dcx, dcy, 3, 1.0);
  dot_k_call("dot_k K=1 [1..3]·[4..6]", ua_dot_k, a, b, 3, 1, 32.0);
  { double g2 = ua_dot2(dcx, dcy, 3);   /* K=2 parity: dot_k(.,.,2) == dot2 bit-for-bit */
    double gk2 = ua_dot_k(dcx, dcy, 3, 2);
    if (g2 != gk2) { printf("FAIL dot_k K=2 == dot2: got %g vs %g\n", gk2, g2); failures++; }
    else printf("ok   dot_k K=2 == dot2 cancel = %g\n", g2); }
  dot_k_call("dot_k K=3 cancel", ua_dot_k, dcx, dcy, 3, 3, 1.0);
  { double gk0 = ua_dot_k(a, b, 3, 0);  /* K<1 treated as 1: K=0 == K=1 bit-for-bit */
    double gk1 = ua_dot_k(a, b, 3, 1);
    if (gk0 != gk1) { printf("FAIL dot_k K=0 == K=1: got %g vs %g\n", gk0, gk1); failures++; }
    else printf("ok   dot_k K=0 == K=1 = %g\n", gk0); }
  dot_k_call("dot_k empty", ua_dot_k, NULL, NULL, 0, 3, 0.0);
  { double g = ua_dot_naive(px, py, 2); /* finite m·(-m) + m·m overflow, no NaN */
    if (isnan(g)) { printf("FAIL naive dot overflow: got NaN want finite/inf\n"); failures++; }
    else printf("ok   naive dot opposite-sign overflow = %g\n", g); }
  { double g = ua_dot2(px, py, 2);
    if (isnan(g)) { printf("FAIL dot2 overflow: got NaN want finite/inf\n"); failures++; }
    else printf("ok   dot2 opposite-sign overflow = %g\n", g); }

  /* ORO sum2 (magnitude-robust compensated): integer-exact, empty 0, recovers
   * 0.1·10 and the cancellation case where Kahan loses the small addends. */
  sum_call("oro [1..4]", ua_sum_oro, small, 4, 10.0);
  sum_call("oro empty", ua_sum_oro, NULL, 0, 0.0);
  sum_call("oro 0.1x10", ua_sum_oro, tenths, 10, 1.0);
  sum_call("oro cancel", ua_sum_oro, cancel, 4, 2.0);

  /* Rump NearSum (correctly rounded): bit-exact with ua_sum_exact on the same
   * cases — integer-exact, empty 0, 0.1·10 → 1.0, cancellation → 2.0. */
  sum_call("near [1..4]", ua_sum_near, small, 4, 10.0);
  sum_call("near empty", ua_sum_near, NULL, 0, 0.0);
  sum_call("near 0.1x10", ua_sum_near, tenths, 10, 1.0);
  sum_call("near cancel", ua_sum_near, cancel, 4, 2.0);

  /* Rump AccSum (faithful): integer-exact, empty 0, 0.1·10 → 1.0 (the exact
   * sum is a float, so faithful picks it), cancellation → 2.0. */
  sum_call("acc [1..4]", ua_sum_acc, small, 4, 10.0);
  sum_call("acc empty", ua_sum_acc, NULL, 0, 0.0);
  sum_call("acc 0.1x10", ua_sum_acc, tenths, 10, 1.0);
  sum_call("acc cancel", ua_sum_acc, cancel, 4, 2.0);

  /* ORO SumK: K=1 naive (integer-exact), K=2 first-order compensated (recovers
   * 0.1·10 and the cancellation case), K<1 treated as 1. */
  sum_k_call("sum_k K=1 [1..4]", ua_sum_k, small, 4, 1, 10.0);
  sum_k_call("sum_k K=2 [1..4]", ua_sum_k, small, 4, 2, 10.0);
  sum_k_call("sum_k K=2 0.1x10", ua_sum_k, tenths, 10, 2, 1.0);
  sum_k_call("sum_k K=2 cancel", ua_sum_k, cancel, 4, 2, 2.0);
  sum_k_call("sum_k K=0 == K=1", ua_sum_k, small, 4, 0, 10.0);

  /* Condition number: 1 on no cancellation, +Inf on an exact-zero sum, 0 on
   * empty, large (≥ 1e19) under catastrophic cancellation. */
  double ones[] = {1.0, 1.0, 1.0, 1.0};
  cond_call("cond no-cancel", ua_condition_number, ones, 4, 1.0);
  cond_call("cond zero-sum", ua_condition_number, (double[]){1.0, -1.0}, 2, INFINITY);
  cond_call("cond empty", ua_condition_number, NULL, 0, 0.0);
  { double c = ua_condition_number(cancel, 4);
    if (c > 1e19) printf("ok   cond cancel = %g (> 1e19)\n", c);
    else { printf("FAIL cond cancel: got %g want > 1e19\n", c); failures++; } }

  sum_call("scaled norm 3-4", ua_norm_scaled,
           (double[]){3.0, 4.0}, 2, 5.0);
  sum_call("scaled norm empty", ua_norm_scaled, NULL, 0, 0.0);
  { double centered[] = {1e10, 1e10 + 1.0, 1e10 + 2.0};
    double paired_centered[] = {2.0, 4.0, 6.0};
    double ss = ua_centered_sum_squares(centered, 3, 1e10 + 1.0);
    double cp = ua_centered_cross_product(centered, paired_centered, 3,
                                          1e10 + 1.0, 4.0);
    if (ss != 2.0 || cp != 4.0) {
      printf("FAIL centered reductions: got (%g, %g) want (2, 4)\n", ss, cp);
      failures++;
    } else printf("ok   centered reductions = (2, 4)\n"); }

  if (failures == 0) { printf("\nAll C ABI tests passed.\n"); return 0; }
  printf("\n%d C ABI test(s) FAILED.\n", failures);
  return 1;
}
