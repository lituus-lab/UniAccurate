#include <stdio.h>
#include "UniAccurate.h"

int main(void) {
  printf("UniAccurate %s\n", ua_version());
  int ns[] = {0, 1, 10, 20, 50, 90, UNIACCURATE_FIB_MAX_N};
  for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); i++)
    printf("fib(%d) = %lld\n", ns[i], ua_fibonacci(ns[i]));
  return 0;
}
