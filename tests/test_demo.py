import pandas as pd
import pytest

from oil_wise.demo import generate_demo_data, write_demo_data


def test_demo_data_is_deterministic() -> None:
    first_prices, first_news = generate_demo_data(days=60, random_seed=7)
    second_prices, second_news = generate_demo_data(days=60, random_seed=7)

    pd.testing.assert_frame_equal(first_prices, second_prices)
    pd.testing.assert_frame_equal(first_news, second_news)
    assert len(first_prices) == 60
    assert len(first_news) == 120
    assert first_prices["close"].gt(0).all()


def test_demo_data_writes_expected_csv_files(tmp_path) -> None:
    price_path, news_path = write_demo_data(tmp_path, days=60)

    assert price_path.exists()
    assert news_path.exists()
    assert {"date", "close"} == set(pd.read_csv(price_path).columns)
    assert {"date", "title", "content"} == set(pd.read_csv(news_path).columns)


def test_demo_requires_enough_history() -> None:
    with pytest.raises(ValueError):
        generate_demo_data(days=20)
