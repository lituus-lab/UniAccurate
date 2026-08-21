# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import array
import math

import pytest
import uniaccurate


def test_scaled_norm_preserves_extremes_and_buffers():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    assert uniaccurate.scaled_norm([largest]) == largest
    assert uniaccurate.scaled_norm(array.array("d", [3.0, 4.0])) == 5.0
    assert math.isnan(uniaccurate.scaled_norm([math.nan]))


def test_scaled_mean_preserves_extremes_and_cancellation():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    assert uniaccurate.scaled_mean([largest, largest]) == largest
    assert uniaccurate.scaled_mean([1e16, 1.0, -1e16]) == 1.0 / 3.0
    with pytest.raises(ValueError):
        uniaccurate.scaled_mean([])


def test_centered_products_support_lists_and_buffers():
    values = [1e10, 1e10 + 1.0, 1e10 + 2.0]
    assert uniaccurate.centered_sum_squares(values, 1e10 + 1.0) == 2.0
    assert uniaccurate.centered_sum_squares(
        array.array("d", values), 1e10 + 1.0) == 2.0
    assert uniaccurate.centered_cross_product(
        values, [2.0, 4.0, 6.0], 1e10 + 1.0, 4.0) == 4.0
    assert uniaccurate.centered_cosine_similarity(
        [1e308, -1e308], [1e308, -1e308], 0.0, 0.0) == pytest.approx(1.0)


def test_centered_products_validate_inputs():
    with pytest.raises(ValueError):
        uniaccurate.centered_cross_product([1.0], [1.0, 2.0], 0.0, 0.0)
    with pytest.raises(TypeError):
        uniaccurate.centered_sum_squares([1.0, "bad"], 0.0)
    with pytest.raises(TypeError):
        uniaccurate.centered_sum_squares([1.0], True)
    with pytest.raises(TypeError):
        uniaccurate.centered_cross_product([1.0], [1.0], 0.0, None)
    with pytest.raises(ValueError):
        uniaccurate.centered_cosine_similarity([], [], 0.0, 0.0)
