"""Stable, headless visualizations for topic and forecasting artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_topic_probabilities(topic_frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot daily topic signals as a stacked area chart."""

    topic_columns = sorted(column for column in topic_frame if column.startswith("topic_"))
    if "date" not in topic_frame.columns or not topic_columns:
        raise ValueError("topic_frame requires date and topic_* columns")
    dates = pd.to_datetime(topic_frame["date"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.stackplot(
        dates,
        *[topic_frame[column].to_numpy() for column in topic_columns],
        labels=topic_columns,
        alpha=0.85,
    )
    axis.set(title="Daily Seeded LDA topic probabilities", xlabel="Date", ylabel="Probability")
    axis.legend(loc="upper left", ncol=2, fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_predictions(
    dates: pd.DatetimeIndex | pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot actual and predicted WTI prices on the held-out period."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(pd.to_datetime(dates), actual, label="Actual WTI", linewidth=2)
    axis.plot(pd.to_datetime(dates), predicted, label="Predicted WTI", linewidth=1.8)
    axis.set(title="WTI close: actual vs predicted", xlabel="Date", ylabel="USD")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_experiment_heatmap(
    leaderboard: pd.DataFrame,
    output_path: str | Path,
    metric: str = "rmse",
) -> Path:
    """Plot a sequence-length × horizon score matrix."""

    required = {"sequence_length", "forecast_horizon", metric}
    if not required.issubset(leaderboard.columns):
        raise ValueError(f"leaderboard requires: {', '.join(sorted(required))}")
    matrix = leaderboard.pivot(
        index="sequence_length",
        columns="forecast_horizon",
        values=metric,
    ).sort_index()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix.to_numpy(), cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set(xlabel="Forecast horizon", ylabel="Sequence length", title=metric.upper())
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label=metric.upper())
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
