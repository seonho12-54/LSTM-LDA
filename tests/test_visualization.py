import numpy as np
import pandas as pd

from oil_wise.visualization import (
    plot_experiment_heatmap,
    plot_predictions,
    plot_topic_probabilities,
)


def test_visualization_artifacts_are_created(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    topics = pd.DataFrame(
        {
            "date": dates,
            "topic_0": np.linspace(0.2, 0.6, 5),
            "topic_1": np.linspace(0.8, 0.4, 5),
        }
    )
    leaderboard = pd.DataFrame(
        {
            "sequence_length": [5, 5, 10, 10],
            "forecast_horizon": [5, 10, 5, 10],
            "rmse": [8.0, 9.0, 7.0, 8.5],
        }
    )

    topic_path = plot_topic_probabilities(topics, tmp_path / "topics.png")
    prediction_path = plot_predictions(
        dates,
        np.arange(5),
        np.arange(5) + 0.5,
        tmp_path / "predictions.png",
    )
    heatmap_path = plot_experiment_heatmap(leaderboard, tmp_path / "heatmap.png")

    assert topic_path.stat().st_size > 0
    assert prediction_path.stat().st_size > 0
    assert heatmap_path.stat().st_size > 0
