import pytest

torch = pytest.importorskip("torch")

from oil_wise.model import PricePredictor  # noqa: E402


def test_price_predictor_returns_one_value_per_sequence() -> None:
    model = PricePredictor(input_size=8, hidden_size=16, num_layers=2)
    batch = torch.randn(4, 20, 8)

    prediction = model(batch)

    assert prediction.shape == (4,)


def test_invalid_model_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        PricePredictor(input_size=0)
