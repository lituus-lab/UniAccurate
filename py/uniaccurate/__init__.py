"""uniaccurate — Python binding over the UniAccurate C library."""
from ._core import fibonacci as _fib_c, version as _version_c

__version__ = _version_c().decode("ascii")
_FIB_MAX_N = 92


def fibonacci(n):
    """fib(n) as int. n in [0, 92]; raises ValueError/TypeError out of range."""
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if not 0 <= n <= _FIB_MAX_N:
        raise ValueError(f"n must be in [0, {_FIB_MAX_N}], got {n}")
    return _fib_c(n)


def version():
    """C library version string."""
    return _version_c().decode("ascii")


__all__ = ["fibonacci", "version", "__version__"]
