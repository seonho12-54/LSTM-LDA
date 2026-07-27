"""Deterministic training and evaluation for the Oil-Wise LSTM."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oil_wise.evaluation import RegressionMetrics, regression_metrics
from oil_wise.model import PricePredictor
from oil_wise.sequences import PreparedSequences


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]

    @property
    def epochs_ran(self) -> int:
        return len(self.train_loss)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: PricePredictor
    history: TrainingHistory
    predictions: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex
    metrics: RegressionMetrics
    device: str


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
    )


def train_price_predictor(
    prepared: PreparedSequences,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    random_seed: int = 42,
    device: str | None = None,
) -> TrainingResult:
    """Train with a tail validation split and evaluate once on the test period."""

    if len(prepared.train) < 3:
        raise ValueError("at least three training sequences are required")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")

    set_reproducible_seed(random_seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(resolved_device)

    validation_size = max(1, int(len(prepared.train) * 0.1))
    train_end = len(prepared.train) - validation_size
    train_loader = _loader(
        prepared.train.features[:train_end],
        prepared.train.targets[:train_end],
        batch_size=batch_size,
        shuffle=True,
        seed=random_seed,
    )
    validation_loader = _loader(
        prepared.train.features[train_end:],
        prepared.train.targets[train_end:],
        batch_size=batch_size,
        shuffle=False,
        seed=random_seed,
    )

    model = PricePredictor(
        input_size=prepared.train.features.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(torch_device)
                batch_targets = batch_targets.to(torch_device)
                batch_losses.append(
                    float(criterion(model(batch_features), batch_targets).detach().cpu())
                )
        validation_loss = float(np.mean(batch_losses))
        validation_losses.append(validation_loss)

        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled_predictions = (
            model(torch.as_tensor(prepared.test.features, dtype=torch.float32).to(torch_device))
            .cpu()
            .numpy()
        )
    predictions = prepared.inverse_targets(scaled_predictions)
    actual = prepared.inverse_targets(prepared.test.targets)
    return TrainingResult(
        model=model,
        history=TrainingHistory(tuple(train_losses), tuple(validation_losses)),
        predictions=predictions,
        actual=actual,
        dates=prepared.test.target_dates,
        metrics=regression_metrics(actual, predictions),
        device=resolved_device,
    )
