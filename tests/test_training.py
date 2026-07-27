import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oil_wise.sequences import prepare_sequences  # noqa: E402
from oil_wise.training import train_price_predictor  # noqa: E402


@pytest.mark.integration
def test_training_smoke_run_returns_unscaled_predictions() -> None:
    random = np.random.default_rng(42)
    index = pd.date_range("2022-01-01", periods=90, freq="D")
    close = 70 + np.sin(np.arange(90) / 6) + random.normal(0, 0.05, 90)
    frame = pd.DataFrame(
        {
            "close": close,
            "topic_0": np.clip(0.5 + np.sin(np.arange(90) / 8) * 0.2, 0, 1),
        },
        index=index,
    )
    prepared = prepare_sequences(
        frame,
        feature_columns=["close", "topic_0"],
        sequence_length=5,
        forecast_horizon=1,
    )

    result = train_price_predictor(
        prepared,
        hidden_size=4,
        num_layers=1,
        epochs=2,
        patience=1,
        batch_size=16,
        device="cpu",
    )

    assert len(result.predictions) == len(prepared.test)
    assert result.history.epochs_ran >= 1
    assert result.metrics.mae >= 0
