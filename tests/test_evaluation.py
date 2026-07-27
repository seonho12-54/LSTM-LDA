import numpy as np
import pytest

from oil_wise.evaluation import regression_metrics


def test_regression_metrics() -> None:
    metrics = regression_metrics(
        actual=np.array([10.0, 12.0, 11.0]),
        predicted=np.array([9.0, 13.0, 10.0]),
    )

    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mape == pytest.approx((0.1 + 1 / 12 + 1 / 11) / 3 * 100)
    assert metrics.directional_accuracy == 100.0


def test_mismatched_arrays_are_rejected() -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))
