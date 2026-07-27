import numpy as np
import pytest

from oil_wise.topic_model import SeededLDATopicModel, build_eta


def test_eta_boosts_only_seeded_topic_word_pairs() -> None:
    eta = build_eta(
        token_to_id={"유가": 0, "전쟁": 1, "시장": 2},
        num_topics=2,
        seed_topics={0: ["유가", "시장"], 1: ["전쟁"]},
    )

    np.testing.assert_array_equal(
        eta,
        np.array([[10.0, 0.01, 10.0], [0.01, 10.0, 0.01]], dtype=np.float32),
    )


def test_invalid_seed_topic_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_eta({"유가": 0}, num_topics=2, seed_topics={2: ["유가"]})


def test_transform_requires_fitted_model() -> None:
    model = SeededLDATopicModel()
    with pytest.raises(RuntimeError):
        model.transform(["유가 상승"])
