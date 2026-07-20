import math
import pytest
import uniaccurate


def test_version():
    assert uniaccurate.version() == "0.0.1"
    assert uniaccurate.__version__ == "0.0.1"


def test_representable_sum_error_is_zero():
    assert uniaccurate.two_sum(1.0, 2.0) == (3.0, 0.0)


def test_lost_addend_recovered():
    s, e = uniaccurate.two_sum(1.0, 2e16)
    assert s == 2e16
    assert e == 1.0


def test_nonfinite_gives_nan_error():
    s, e = uniaccurate.two_sum(float("inf"), 1.0)
    assert math.isinf(s)
    assert math.isnan(e)


def test_int_inputs_promoted():
    assert uniaccurate.two_sum(1, 2) == (3.0, 0.0)


def test_non_numeric_raises():
    with pytest.raises(TypeError):
        uniaccurate.two_sum("x", 2.0)
    with pytest.raises(TypeError):
        uniaccurate.two_sum(1.0, None)
    with pytest.raises(TypeError):
        uniaccurate.two_sum(True, 2.0)
