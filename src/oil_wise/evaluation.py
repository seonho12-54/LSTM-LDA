"""Regression metrics used to compare LSTM experiment settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    """Calculate level and direction metrics with safe zero handling."""

    y_true = np.asarray(actual, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("actual and predicted must be non-empty and equally sized")

    errors = y_true - y_pred
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    if len(y_true) < 2:
        directional_accuracy = float("nan")
    else:
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
        directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    return RegressionMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        mape=mape,
        directional_accuracy=directional_accuracy,
    )
