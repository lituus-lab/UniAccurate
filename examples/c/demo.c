#include <stdio.h>
#include <stddef.h>
#include "UniAccurate.h"

int main(void) {
  printf("UniAccurate %s\n", ua_version());
  double as[] = {1.0, 1.0, 1e20};
  double bs[] = {2.0, 2e16, 1.0};
  for (size_t i = 0; i < sizeof(as) / sizeof(as[0]); i++) {
    double s, e;
    ua_two_sum(as[i], bs[i], &s, &e);
    printf("ua_two_sum(%g, %g) = (%g, %g)\n", as[i], bs[i], s, e);
  }
  return 0;
}
