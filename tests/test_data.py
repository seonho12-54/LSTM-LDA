from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oil_wise.data import (
    DataSchemaError,
    aggregate_daily_documents,
    load_news_csv,
    load_price_csv,
    merge_price_and_topics,
)


def test_news_loader_combines_title_and_body(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "날짜": ["2024-01-01 09:00", "2024-01-01 17:00"],
            "제목": ["OPEC 감산", "WTI 상승"],
            "본문": ["공급 감소 전망", "재고 감소 영향"],
        }
    ).to_csv(path, index=False, encoding="cp949")

    news = load_news_csv(path)
    daily = aggregate_daily_documents(news)

    assert news.loc[0, "text"] == "OPEC 감산 공급 감소 전망"
    assert len(daily) == 1
    assert daily.loc[0, "article_count"] == 2


def test_price_loader_parses_currency_and_commas(tmp_path: Path) -> None:
    path = tmp_path / "wti.csv"
    pd.DataFrame(
        {"Date": ["2024-01-01", "2024-01-02"], "Close": ["$71.25", "1,070.50"]}
    ).to_csv(path, index=False)

    prices = load_price_csv(path)

    assert prices["close"].tolist() == [71.25, 1070.5]


def test_missing_price_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"value": [1]}).to_csv(path, index=False)

    with pytest.raises(DataSchemaError):
        load_price_csv(path)


def test_price_topic_merge_adds_market_features() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"close": np.linspace(70, 79, 10)}, index=index)
    topics = pd.DataFrame(
        {
            "date": index,
            "topic_0": np.linspace(0.1, 0.3, 10),
            "topic_1": np.linspace(0.9, 0.7, 10),
        }
    )

    result = merge_price_and_topics(prices, topics)

    assert {"return_1d", "volatility_5d", "topic_0", "topic_1"}.issubset(result.columns)
    assert result.isna().sum().sum() == 0
