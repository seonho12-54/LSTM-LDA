import pytest

from oil_wise.config import TOPIC_LABELS, TOPIC_SEEDS, ExperimentConfig


def test_seven_topic_contract_is_complete() -> None:
    assert sorted(TOPIC_LABELS) == list(range(7))
    assert sorted(TOPIC_SEEDS) == list(range(7))
    assert all(TOPIC_SEEDS[topic_id] for topic_id in TOPIC_SEEDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_topics", 1),
        ("sequence_length", 1),
        ("forecast_horizon", 0),
        ("hidden_size", 0),
        ("num_layers", 0),
        ("dropout", 1.0),
        ("learning_rate", 0),
        ("epochs", 0),
    ],
)
def test_invalid_experiment_settings_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{field: value})
