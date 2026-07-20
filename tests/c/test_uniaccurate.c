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

int main(void) {
  two_sum("1 + 2", 1.0, 2.0, 3.0, 0.0);
  two_sum("1 + 2e16", 1.0, 2e16, 2e16, 1.0);
  { double s, e;
    ua_two_sum(INFINITY, 1.0, &s, &e);
    if (isinf(s) && isnan(e)) printf("ok   inf + 1 = (inf, nan)\n");
    else { printf("FAIL inf + 1: got (%g, %g) want (inf, nan)\n", s, e); failures++; } }
  check_str("version", ua_version(), UNIACCURATE_VERSION);

  if (failures == 0) { printf("\nAll C ABI tests passed.\n"); return 0; }
  printf("\n%d C ABI test(s) FAILED.\n", failures);
  return 1;
}
