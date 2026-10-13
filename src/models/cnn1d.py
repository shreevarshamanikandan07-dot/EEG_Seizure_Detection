"""1D Deep Convolutional Neural Network for Raw Multichannel EEG Windows.

Processes continuous spatio-temporal EEG tensors (Batch, Channels, Time)
using hierarchical temporal convolutional representations, batch normalization,
GELU activations, and adaptive temporal pooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Temporal convolution block with Batch Normalization, GELU, Pooling, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class EEG1DCNN(nn.Module):
    """Multichannel 1D CNN for EEG Seizure Segment Classification."""

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 2,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[int, ...] = (7, 7, 5),
        fc_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Hierarchical Feature Extraction
        blocks = []
        cur_in = in_channels
        for out_c, k_size in zip(conv_channels, kernel_sizes):
            pad = k_size // 2
            blocks.append(
                ConvBlock1D(
                    in_channels=cur_in,
                    out_channels=out_c,
                    kernel_size=k_size,
                    padding=pad,
                    pool_size=2,
                    dropout=dropout * 0.7,
                )
            )
            cur_in = out_c
        self.feature_extractor = nn.Sequential(*blocks)

        # Global Spatio-Temporal Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Decision Network
        last_channels = conv_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        features = self.feature_extractor(x)
        pooled = self.global_pool(features).squeeze(-1)  # (batch_size, last_channels)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute seizure prediction probabilities.

        Args:
            x: Input tensor (B, C, T).

        Returns:
            torch.Tensor: Positive class (seizure) probability of shape (B,).
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

    def get_num_parameters(self) -> int:
        """Return count of trainable weights."""
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
            },
            str(p),
        )

    def load(self, filepath: str | Path, map_location: str = "cpu") -> EEG1DCNN:
        """Load model weights from checkpoint."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        ckpt = torch.load(str(p), map_location=map_location, weights_only=True)
        self.load_state_dict(ckpt["state_dict"])
        self.eval()
        return self
