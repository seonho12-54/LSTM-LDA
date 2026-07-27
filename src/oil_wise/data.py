"""Load, validate, and align news articles with WTI price observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

DATE_CANDIDATES = ("date", "datetime", "published_at", "작성일", "등록일", "날짜", "일자")
TITLE_CANDIDATES = ("title", "headline", "제목", "기사제목")
BODY_CANDIDATES = ("content", "body", "article", "description", "본문", "내용", "기사내용")
PRICE_CANDIDATES = ("close", "price", "wti", "종가", "가격")


class DataSchemaError(ValueError):
    """Raised when an input file does not match the expected data contract."""


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def _read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataSchemaError(f"{source.name}: unsupported text encoding") from last_error


def _resolve_column(
    columns: Iterable[str],
    candidates: Sequence[str],
    requested: str | None = None,
) -> str | None:
    available = list(columns)
    if requested is not None:
        normalized = requested.strip().lower()
        return normalized if normalized in available else None
    for candidate in candidates:
        for column in available:
            if column == candidate or candidate in column:
                return column
    return None


def load_news_csv(
    path: str | Path,
    date_column: str | None = None,
    text_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load news with a normalized date and combined text column."""

    frame = normalize_columns(_read_csv_with_fallback(path))
    resolved_date = _resolve_column(frame.columns, DATE_CANDIDATES, date_column)
    if resolved_date is None:
        raise DataSchemaError("news data requires a recognizable date column")

    if text_columns is None:
        resolved_text = [
            column
            for column in (
                _resolve_column(frame.columns, TITLE_CANDIDATES),
                _resolve_column(frame.columns, BODY_CANDIDATES),
            )
            if column is not None
        ]
    else:
        resolved_text = [column.strip().lower() for column in text_columns]
    resolved_text = list(dict.fromkeys(resolved_text))
    missing = [column for column in resolved_text if column not in frame.columns]
    if missing:
        raise DataSchemaError(f"news text columns not found: {', '.join(missing)}")
    if not resolved_text:
        raise DataSchemaError("news data requires at least one title or body column")

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[resolved_date], errors="coerce").dt.normalize()
    text_parts = [frame[column].fillna("").astype(str).str.strip() for column in resolved_text]
    result["text"] = pd.concat(text_parts, axis=1).agg(" ".join, axis=1).str.strip()
    result = result.dropna(subset=["date"])
    result = result[result["text"].str.len() > 0]
    return result.sort_values("date").reset_index(drop=True)


def load_price_csv(
    path: str | Path,
    date_column: str | None = None,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Load WTI prices and return a date-indexed close series."""

    frame = normalize_columns(_read_csv_with_fallback(path))
    resolved_date = _resolve_column(frame.columns, DATE_CANDIDATES, date_column)
    resolved_price = _resolve_column(frame.columns, PRICE_CANDIDATES, price_column)
    if resolved_date is None or resolved_price is None:
        raise DataSchemaError("price data requires date and close/price columns")

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[resolved_date], errors="coerce").dt.normalize()
    raw_price = (
        frame[resolved_price]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    result["close"] = pd.to_numeric(raw_price, errors="coerce")
    result = result.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    if result.empty:
        raise DataSchemaError("price data contains no usable observations")
    return result.set_index("date").sort_index().astype({"close": "float64"})


def aggregate_daily_documents(news: pd.DataFrame) -> pd.DataFrame:
    """Combine all news published on the same day into one document."""

    if not {"date", "text"}.issubset(news.columns):
        raise DataSchemaError("news frame must contain date and text columns")
    result = news.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["text"] = result["text"].fillna("").astype(str).str.strip()
    result = result.dropna(subset=["date"])
    result = result[result["text"].str.len() > 0]
    return (
        result.groupby("date", as_index=False)
        .agg(text=("text", " ".join), article_count=("text", "size"))
        .sort_values("date")
        .reset_index(drop=True)
    )


def merge_price_and_topics(
    prices: pd.DataFrame,
    topic_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Align topic probabilities to trading days without leaking future news."""

    if "close" not in prices.columns:
        raise DataSchemaError("prices must contain a close column")
    topic_columns = [
        column for column in topic_probabilities.columns if column.startswith("topic_")
    ]
    if not topic_columns:
        raise DataSchemaError("topic frame must contain topic_0 ... topic_n columns")

    topics = topic_probabilities.copy()
    if "date" in topics.columns:
        topics["date"] = pd.to_datetime(topics["date"], errors="coerce").dt.normalize()
        topics = topics.dropna(subset=["date"]).set_index("date")
    if not isinstance(topics.index, pd.DatetimeIndex):
        raise DataSchemaError("topic frame requires a DatetimeIndex or date column")

    result = prices.sort_index().join(topics[topic_columns], how="left")
    result[topic_columns] = result[topic_columns].fillna(0.0)
    result["return_1d"] = result["close"].pct_change()
    result["volatility_5d"] = result["return_1d"].rolling(5).std()
    return result.dropna()
