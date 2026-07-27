"""Experiment grid matching the sequence/horizon study shown in the poster."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from oil_wise.sequences import prepare_sequences


@dataclass(frozen=True, slots=True)
class ExperimentGrid:
    leaderboard: pd.DataFrame
    runs: dict[tuple[int, int], Any]

    @property
    def best_setting(self) -> dict[str, float | int]:
        if self.leaderboard.empty:
            raise ValueError("experiment grid is empty")
        return self.leaderboard.iloc[0].to_dict()


def run_experiment_grid(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    sequence_lengths: Iterable[int] = (5, 10, 20, 30),
    forecast_horizons: Iterable[int] = (5, 10, 20, 30),
    train_ratio: float = 0.8,
    trainer: Callable[..., Any] | None = None,
    **training_options: Any,
) -> ExperimentGrid:
    """Train every requested sequence/horizon pair and rank results by RMSE."""

    if trainer is None:
        try:
            from oil_wise.training import train_price_predictor
        except ImportError as exc:
            raise ImportError("experiment training requires: pip install -e .[lstm]") from exc

        trainer = train_price_predictor

    rows: list[dict[str, float | int]] = []
    runs: dict[tuple[int, int], Any] = {}
    for sequence_length in sequence_lengths:
        for horizon in forecast_horizons:
            prepared = prepare_sequences(
                frame,
                feature_columns=feature_columns,
                target_column="close",
                sequence_length=sequence_length,
                forecast_horizon=horizon,
                train_ratio=train_ratio,
            )
            result = trainer(prepared, **training_options)
            runs[(sequence_length, horizon)] = result
            rows.append(
                {
                    "sequence_length": sequence_length,
                    "forecast_horizon": horizon,
                    **result.metrics.as_dict(),
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["rmse", "mae", "sequence_length", "forecast_horizon"]
    )
    return ExperimentGrid(leaderboard=leaderboard.reset_index(drop=True), runs=runs)
