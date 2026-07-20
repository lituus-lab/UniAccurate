# cython: language_level=3
cdef extern from "UniAccurate.h":
    const char *ua_version()
    long long ua_fibonacci(int n)


def fibonacci(int n):
    """Raw C call (no domain check). Use uniaccurate.fibonacci."""
    return ua_fibonacci(n)


def version():
    return ua_version()
