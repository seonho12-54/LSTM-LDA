"""PyTorch LSTM architecture used by Oil-Wise AI."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise ImportError("LSTM support requires: pip install -e .[lstm]") from exc


class PricePredictor(nn.Module):
    """Many-to-one LSTM regressor for future WTI closing prices."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(features)
        last_hidden = self.dropout(sequence_output[:, -1, :])
        return self.output(last_hidden).squeeze(-1)
