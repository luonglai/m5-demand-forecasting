import math

import numpy as np
import pytest

from m5fc.baseline import classify_demand
from m5fc.evaluate import bias, wmape
from m5fc.inventory import safety_stock_factor


def test_wmape_and_bias():
    y = np.array([10.0, 0.0, 5.0, 5.0])
    f = np.array([12.0, 1.0, 4.0, 5.0])
    assert wmape(y, f) == pytest.approx((2 + 1 + 1 + 0) / 20)
    assert bias(y, f) == pytest.approx((2 + 1 - 1 + 0) / 20)


def test_metrics_with_no_demand_return_nan():
    assert math.isnan(wmape(np.zeros(3), np.ones(3)))


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.array([5, 6, 5, 7, 6, 5, 6, 5]), "smooth"),
        (np.array([1, 20, 2, 30, 1, 25, 2, 40]), "erratic"),
        (np.array([0, 0, 3, 0, 0, 3, 0, 0, 4, 0]), "intermittent"),
        (np.array([0, 0, 1, 0, 0, 30, 0, 0, 2, 0]), "lumpy"),
        (np.zeros(10), "no_demand"),
    ],
)
def test_classify_demand(y, expected):
    assert classify_demand(y)[2] == expected


def test_safety_stock_factor():
    assert safety_stock_factor(0.95, 1) == pytest.approx(1.645, abs=1e-3)
    assert safety_stock_factor(0.95, 4) == pytest.approx(2 * 1.645, abs=1e-3)
