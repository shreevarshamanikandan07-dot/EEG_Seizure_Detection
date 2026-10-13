"""Hybrid Spatio-Temporal CNN-LSTM Architecture for EEG Seizure Detection.

Combines 1D temporal convolutions for localized morphological transient extraction
with Long Short-Term Memory (LSTM) recurrent networks to model sequential epileptic evolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_CNNLSTM(nn.Module):
    """Hybrid Spatio-Temporal CNN-LSTM Network."""

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 2,
        cnn_channels: Tuple[int, ...] = (32, 64),
        kernel_size: int = 7,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Stage 1: Convolutional feature extraction
        conv_layers = []
        cur_in = in_channels
        for out_c in cnn_channels:
            pad = kernel_size // 2
            conv_layers.extend([
                nn.Conv1d(cur_in, out_c, kernel_size=kernel_size, padding=pad, bias=False),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=4),  # Temporal downsampling 1024 -> 256 -> 64
                nn.Dropout(dropout * 0.5),
            ])
            cur_in = out_c
        self.conv_net = nn.Sequential(*conv_layers)

        # Stage 2: Recurrent sequence modeling
        last_cnn_channels = cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=last_cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Stage 3: Fully connected decision layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # 1. CNN feature extraction: (B, in_channels, seq_len) -> (B, last_cnn_channels, T_reduced)
        cnn_feats = self.conv_net(x)

        # 2. Transpose to sequence format: (B, T_reduced, last_cnn_channels)
        seq_feats = cnn_feats.permute(0, 2, 1)

        # 3. LSTM sequential tracking
        lstm_out, _ = self.lstm(seq_feats)  # (B, T_reduced, lstm_hidden)

        # 4. Temporal representation pooling (mean over sequence dimension)
        context = torch.mean(lstm_out, dim=1)  # (B, lstm_hidden)

        # 5. Classification logits
        logits = self.classifier(context)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return positive class probability tensor of shape (B,)."""
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

    def get_num_parameters(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, filepath: str | Path) -> None:
        """Save model checkpoint dictionary."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "lstm_hidden": self.lstm_hidden,
                "lstm_layers": self.lstm_layers,
            },
            str(p),
        )

    def load(self, filepath: str | Path, map_location: str = "cpu") -> EEG_CNNLSTM:
        """Load checkpoint weights."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        ckpt = torch.load(str(p), map_location=map_location, weights_only=True)
        self.load_state_dict(ckpt["state_dict"])
        self.eval()
        return self
