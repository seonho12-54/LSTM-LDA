import numpy as np
import pandas as pd
import pytest

from oil_wise.sequences import create_sequences, prepare_sequences


def test_forecast_horizon_targets_the_correct_future_row() -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    targets = np.arange(10, dtype=np.float32)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    result = create_sequences(features, targets, dates, sequence_length=3, forecast_horizon=2)

    assert result.features.shape == (6, 3, 2)
    assert result.targets[0] == 4
    assert result.target_dates[0] == pd.Timestamp("2024-01-05")


def test_scalers_are_fitted_only_on_training_rows() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {
            "close": np.r_[np.arange(32), np.arange(1000, 1008)],
            "topic_0": np.r_[np.arange(32), np.arange(1000, 1008)],
        },
        index=index,
    )

    prepared = prepare_sequences(
        frame,
        feature_columns=["close", "topic_0"],
        sequence_length=5,
        forecast_horizon=1,
        train_ratio=0.8,
    )

    assert prepared.feature_scaler.mean_[0] == pytest.approx(np.arange(32).mean())
    assert prepared.target_scaler.mean_[0] == pytest.approx(np.arange(32).mean())
    assert prepared.train.target_dates.max() < prepared.split_date
    assert prepared.test.target_dates.min() >= prepared.split_date


def test_short_series_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_sequences(
            np.ones((5, 2)),
            np.ones(5),
            pd.date_range("2024-01-01", periods=5),
            sequence_length=4,
            forecast_horizon=2,
        )
