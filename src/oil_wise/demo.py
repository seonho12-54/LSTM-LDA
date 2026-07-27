"""Deterministic synthetic data for an end-to-end Oil-Wise quickstart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TOPIC_TEMPLATES = (
    "정부 정책 발표와 원유 시장 안정 대책",
    "원유 수출 증가와 글로벌 경제 회복 전망",
    "에너지 전환과 탄소 배출 환경 규제 강화",
    "정유 산업 기술 혁신과 생산 효율 개선",
    "석유 시설 안전 점검과 공급망 위험 관리",
    "산유국 외교 협력과 국제 원유 협상 진행",
    "국제 정세 긴장과 전쟁 우려로 유가 변동",
)


def generate_demo_data(
    days: int = 800,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic business-day WTI prices and Korean news articles."""

    if days < 60:
        raise ValueError("days must be at least 60")
    random = np.random.default_rng(random_seed)
    dates = pd.bdate_range("2021-01-04", periods=days)
    topic_probabilities = random.dirichlet(np.ones(7) * 1.5, size=days)
    topic_effects = np.array([0.02, 0.07, -0.06, 0.03, -0.04, 0.05, -0.08])
    seasonal = 0.08 * np.sin(np.arange(days) * 2 * np.pi / 120)
    returns = (
        0.0003
        + topic_probabilities @ topic_effects * 0.02
        + seasonal * 0.01
        + random.normal(0, 0.012, days)
    )
    close = 62.0 * np.exp(np.cumsum(returns))
    prices = pd.DataFrame({"date": dates, "close": close.round(4)})

    article_rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        dominant = np.argsort(topic_probabilities[index])[-2:][::-1]
        for rank, topic_id in enumerate(dominant):
            article_rows.append(
                {
                    "date": date + pd.Timedelta(hours=9 + rank * 5),
                    "title": TOPIC_TEMPLATES[topic_id],
                    "content": (
                        f"{TOPIC_TEMPLATES[topic_id]}. 시장 관계자는 향후 가격과 수급 변화를 "
                        "지켜봐야 한다고 분석했다."
                    ),
                }
            )
    news = pd.DataFrame(article_rows)
    return prices, news


def write_demo_data(
    output_directory: str | Path,
    days: int = 800,
    random_seed: int = 42,
) -> tuple[Path, Path]:
    """Write demo CSV files and return the price/news paths."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prices, news = generate_demo_data(days=days, random_seed=random_seed)
    price_path = output / "wti_prices.csv"
    news_path = output / "oil_news.csv"
    prices.to_csv(price_path, index=False, encoding="utf-8-sig")
    news.to_csv(news_path, index=False, encoding="utf-8-sig")
    return price_path, news_path
