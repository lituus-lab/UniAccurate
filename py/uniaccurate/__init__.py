"""uniaccurate — Python binding over the UniAccurate C library."""
from ._core import two_sum as _two_sum_c, version as _version_c

__version__ = _version_c().decode("ascii")


def two_sum(a, b):
    """Error-free sum: returns (s, e) with a + b == s + e exactly.
    s = fl(a + b) (IEEE-754); e carries the rounding error. Non-finite a or b
    gives s = a + b (IEEE) and e = NaN. Raises TypeError on non-numeric input.
    """
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        raise TypeError(f"a must be a number, got {type(a).__name__}")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        raise TypeError(f"b must be a number, got {type(b).__name__}")
    return _two_sum_c(float(a), float(b))


def version():
    """C library version string."""
    return _version_c().decode("ascii")


__all__ = ["two_sum", "version", "__version__"]
