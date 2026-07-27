from pathlib import Path

import pandas as pd

from oil_wise.cli import build_parser, main


def test_parser_uses_poster_experiment_defaults() -> None:
    arguments = build_parser().parse_args(
        [
            "grid",
            "--prices",
            "prices.csv",
            "--topics",
            "topics.csv",
            "--output",
            "artifacts",
        ]
    )

    assert arguments.sequence_lengths == [5, 10, 20, 30]
    assert arguments.horizons == [5, 10, 20, 30]


def test_demo_command_creates_quickstart_data(tmp_path: Path) -> None:
    exit_code = main(["demo", "--output", str(tmp_path), "--days", "60"])

    assert exit_code == 0
    assert len(pd.read_csv(tmp_path / "wti_prices.csv")) == 60
    assert len(pd.read_csv(tmp_path / "oil_news.csv")) == 120
