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

  if (failures == 0) { printf("\nAll C ABI tests passed.\n"); return 0; }
  printf("\n%d C ABI test(s) FAILED.\n", failures);
  return 1;
}
