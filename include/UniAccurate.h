#ifndef UNIACCURATE_H
#define UNIACCURATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UNIACCURATE_VERSION_MAJOR 0
#define UNIACCURATE_VERSION_MINOR 0
#define UNIACCURATE_VERSION_PATCH 1
#define UNIACCURATE_VERSION "0.0.1"

#define UNIACCURATE_VERSION_AT_LEAST(ma, mi, pa) \
  ((UNIACCURATE_VERSION_MAJOR > (ma)) || \
   (UNIACCURATE_VERSION_MAJOR == (ma) && UNIACCURATE_VERSION_MINOR > (mi)) || \
   (UNIACCURATE_VERSION_MAJOR == (ma) && UNIACCURATE_VERSION_MINOR == (mi) && \
    UNIACCURATE_VERSION_PATCH >= (pa)))

/* Largest n with ua_fibonacci(n) fitting in long long (int64). */
#define UNIACCURATE_FIB_MAX_N 92

/* Static version string; do not free. */
const char *ua_version(void);

/* fibonacci(n), n clamped to [0, UNIACCURATE_FIB_MAX_N].
 * n < 0 -> 0; n > UNIACCURATE_FIB_MAX_N -> fibonacci(UNIACCURATE_FIB_MAX_N).
 * Never raises. Single-threaded, reentrant. */
long long ua_fibonacci(int n);

#ifdef __cplusplus
}
#endif

#endif /* UNIACCURATE_H */
