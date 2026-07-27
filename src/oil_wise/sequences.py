"""Chronological scaling and sequence creation for multivariate LSTM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """A set of LSTM windows with target values and their dates."""

    features: np.ndarray
    targets: np.ndarray
    target_dates: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.targets)


@dataclass(frozen=True, slots=True)
class PreparedSequences:
    """Train/test sequences and the scalers fitted only on training rows."""

    train: SequenceBatch
    test: SequenceBatch
    feature_scaler: Any
    target_scaler: Any
    feature_columns: tuple[str, ...]
    target_column: str
    split_date: pd.Timestamp

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return self.target_scaler.inverse_transform(array).reshape(-1)


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    sequence_length: int,
    forecast_horizon: int,
) -> SequenceBatch:
    """Create windows where each target occurs exactly `forecast_horizon` steps later."""

    x_values = np.asarray(features, dtype=np.float32)
    y_values = np.asarray(targets, dtype=np.float32).reshape(-1)
    date_index = pd.DatetimeIndex(dates)
    if x_values.ndim != 2:
        raise ValueError("features must have shape [time, features]")
    if len(x_values) != len(y_values) or len(y_values) != len(date_index):
        raise ValueError("features, targets, and dates must have the same length")
    if sequence_length < 2 or forecast_horizon < 1:
        raise ValueError("sequence_length >= 2 and forecast_horizon >= 1 are required")

    sample_count = len(x_values) - sequence_length - forecast_horizon + 1
    if sample_count < 1:
        raise ValueError("series is too short for the requested sequence and horizon")

    windows = np.stack(
        [x_values[start : start + sequence_length] for start in range(sample_count)]
    )
    target_positions = np.arange(sample_count) + sequence_length + forecast_horizon - 1
    return SequenceBatch(
        features=windows,
        targets=y_values[target_positions],
        target_dates=date_index[target_positions],
    )


def prepare_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "close",
    sequence_length: int = 20,
    forecast_horizon: int = 5,
    train_ratio: float = 0.8,
) -> PreparedSequences:
    """Fit scalers on past rows and split sequence targets chronologically."""

    if not 0.5 <= train_ratio < 1:
        raise ValueError("train_ratio must be in [0.5, 1)")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must use a DatetimeIndex")

    columns = tuple(feature_columns)
    missing = [column for column in (*columns, target_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"columns not found: {', '.join(missing)}")

    ordered = frame.sort_index().dropna(subset=[*columns, target_column])
    split_position = int(len(ordered) * train_ratio)
    minimum_train = sequence_length + forecast_horizon
    if split_position < minimum_train or len(ordered) - split_position < 1:
        raise ValueError("data is too short for the requested chronological split")

    feature_scaler = StandardScaler().fit(ordered.iloc[:split_position][list(columns)])
    target_scaler = StandardScaler().fit(
        ordered.iloc[:split_position][[target_column]].to_numpy()
    )
    scaled_features = feature_scaler.transform(ordered[list(columns)])
    scaled_targets = target_scaler.transform(ordered[[target_column]].to_numpy()).reshape(-1)
    all_sequences = create_sequences(
        scaled_features,
        scaled_targets,
        ordered.index,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    split_date = ordered.index[split_position]
    is_train = all_sequences.target_dates < split_date
    train = SequenceBatch(
        features=all_sequences.features[is_train],
        targets=all_sequences.targets[is_train],
        target_dates=all_sequences.target_dates[is_train],
    )
    test = SequenceBatch(
        features=all_sequences.features[~is_train],
        targets=all_sequences.targets[~is_train],
        target_dates=all_sequences.target_dates[~is_train],
    )
    if len(train) == 0 or len(test) == 0:
        raise ValueError("chronological split produced an empty train or test set")

    return PreparedSequences(
        train=train,
        test=test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=columns,
        target_column=target_column,
        split_date=split_date,
    )
