"""Command-line entry points for data generation, topic modelling, and forecasting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from oil_wise.data import (
    aggregate_daily_documents,
    load_news_csv,
    load_price_csv,
    merge_price_and_topics,
)
from oil_wise.demo import write_demo_data


def _write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def _training_frame(price_path: str, topic_path: str) -> tuple[pd.DataFrame, list[str]]:
    prices = load_price_csv(price_path)
    topics = pd.read_csv(topic_path)
    frame = merge_price_and_topics(prices, topics)
    topic_columns = sorted(column for column in frame.columns if column.startswith("topic_"))
    feature_columns = ["close", *topic_columns, "return_1d", "volatility_5d"]
    return frame, feature_columns


def command_demo(arguments: argparse.Namespace) -> int:
    price_path, news_path = write_demo_data(
        arguments.output,
        days=arguments.days,
        random_seed=arguments.seed,
    )
    print(f"WTI prices: {price_path}")
    print(f"Korean news: {news_path}")
    return 0


def command_topics(arguments: argparse.Namespace) -> int:
    from oil_wise.topic_model import SeededLDATopicModel

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    news = load_news_csv(arguments.news)
    daily = aggregate_daily_documents(news)
    model = SeededLDATopicModel(
        num_topics=arguments.num_topics,
        passes=arguments.passes,
        iterations=arguments.iterations,
        random_state=arguments.seed,
        no_below=arguments.no_below,
    ).fit(daily["text"].tolist())
    topics = model.transform_daily(daily)
    topic_path = output / "daily_topic_probabilities.csv"
    topics.to_csv(topic_path, index=False, encoding="utf-8-sig")

    keywords = {
        str(topic_id): [
            {"word": word, "probability": float(probability)}
            for word, probability in topic_words
        ]
        for topic_id, topic_words in model.top_keywords(top_n=arguments.top_words).items()
    }
    _write_json(output / "topic_keywords.json", keywords)
    model.model.save(str(output / "seeded_lda.model"))
    model.dictionary.save(str(output / "seeded_lda.dictionary"))
    print(f"Daily topics: {topic_path}")
    print(f"Tokenizer backend: {model.tokenizer.backend}")
    return 0


def command_train(arguments: argparse.Namespace) -> int:
    import joblib
    import torch

    from oil_wise.sequences import prepare_sequences
    from oil_wise.training import train_price_predictor

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, feature_columns = _training_frame(arguments.prices, arguments.topics)
    prepared = prepare_sequences(
        frame,
        feature_columns=feature_columns,
        sequence_length=arguments.sequence_length,
        forecast_horizon=arguments.horizon,
        train_ratio=arguments.train_ratio,
    )
    result = train_price_predictor(
        prepared,
        hidden_size=arguments.hidden_size,
        num_layers=arguments.num_layers,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        random_seed=arguments.seed,
    )

    predictions = pd.DataFrame(
        {
            "date": result.dates,
            "actual_close": result.actual,
            "predicted_close": result.predictions,
        }
    )
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    _write_json(
        output / "metrics.json",
        {
            **result.metrics.as_dict(),
            "epochs_ran": result.history.epochs_ran,
            "device": result.device,
            "sequence_length": arguments.sequence_length,
            "forecast_horizon": arguments.horizon,
        },
    )
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "input_size": prepared.train.features.shape[-1],
            "hidden_size": arguments.hidden_size,
            "num_layers": arguments.num_layers,
            "dropout": arguments.dropout,
            "feature_columns": feature_columns,
        },
        output / "price_predictor.pt",
    )
    joblib.dump(prepared.feature_scaler, output / "feature_scaler.joblib")
    joblib.dump(prepared.target_scaler, output / "target_scaler.joblib")
    print(f"Metrics: {output / 'metrics.json'}")
    print(f"Predictions: {output / 'predictions.csv'}")
    return 0


def command_grid(arguments: argparse.Namespace) -> int:
    from oil_wise.experiments import run_experiment_grid

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, feature_columns = _training_frame(arguments.prices, arguments.topics)
    result = run_experiment_grid(
        frame,
        feature_columns=feature_columns,
        sequence_lengths=arguments.sequence_lengths,
        forecast_horizons=arguments.horizons,
        train_ratio=arguments.train_ratio,
        hidden_size=arguments.hidden_size,
        num_layers=arguments.num_layers,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        random_seed=arguments.seed,
    )
    result.leaderboard.to_csv(output / "leaderboard.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "best_setting.json", result.best_setting)
    print(result.leaderboard.to_string(index=False))
    return 0


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prices", required=True, help="WTI CSV with date and close columns")
    parser.add_argument("--topics", required=True, help="daily topic-probability CSV")
    parser.add_argument("--output", required=True, help="artifact output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oil-wise",
        description="Seeded LDA news signals and LSTM WTI price forecasting",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo_parser = commands.add_parser("demo", help="generate deterministic sample data")
    demo_parser.add_argument("--output", default="data/demo")
    demo_parser.add_argument("--days", type=int, default=800)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.set_defaults(handler=command_demo)

    topic_parser = commands.add_parser("topics", help="train Seeded LDA and export daily topics")
    topic_parser.add_argument("--news", required=True)
    topic_parser.add_argument("--output", required=True)
    topic_parser.add_argument("--num-topics", type=int, default=7)
    topic_parser.add_argument("--passes", type=int, default=30)
    topic_parser.add_argument("--iterations", type=int, default=100)
    topic_parser.add_argument("--no-below", type=int, default=2)
    topic_parser.add_argument("--top-words", type=int, default=10)
    topic_parser.add_argument("--seed", type=int, default=42)
    topic_parser.set_defaults(handler=command_topics)

    train_parser = commands.add_parser("train", help="train and evaluate one LSTM setting")
    _add_training_options(train_parser)
    train_parser.add_argument("--sequence-length", type=int, default=20)
    train_parser.add_argument("--horizon", type=int, default=5)
    train_parser.set_defaults(handler=command_train)

    grid_parser = commands.add_parser("grid", help="compare sequence and forecast horizons")
    _add_training_options(grid_parser)
    grid_parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
    )
    grid_parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
    )
    grid_parser.set_defaults(handler=command_grid)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
