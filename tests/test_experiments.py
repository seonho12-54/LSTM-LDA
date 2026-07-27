from types import SimpleNamespace

import numpy as np
import pandas as pd

from oil_wise.evaluation import RegressionMetrics
from oil_wise.experiments import run_experiment_grid


def test_experiment_grid_ranks_lowest_rmse_first() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    frame = pd.DataFrame(
        {
            "close": 60 + np.sin(np.arange(120) / 10),
            "topic_0": np.linspace(0.1, 0.9, 120),
        },
        index=index,
    )

    def fake_trainer(prepared, **_):
        sequence_length = prepared.train.features.shape[1]
        score = float(abs(sequence_length - 10))
        return SimpleNamespace(
            metrics=RegressionMetrics(score, score, score, 50.0),
        )

    result = run_experiment_grid(
        frame,
        feature_columns=["close", "topic_0"],
        sequence_lengths=[5, 10],
        forecast_horizons=[1, 2],
        trainer=fake_trainer,
    )

    assert len(result.leaderboard) == 4
    assert result.best_setting["sequence_length"] == 10
    assert set(result.runs) == {(5, 1), (5, 2), (10, 1), (10, 2)}
