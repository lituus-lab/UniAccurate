// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 lituus-lab
#include <stdio.h>
#include <string.h>
#include <math.h>
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

  if (failures == 0) { printf("\nAll C ABI tests passed.\n"); return 0; }
  printf("\n%d C ABI test(s) FAILED.\n", failures);
  return 1;
}
