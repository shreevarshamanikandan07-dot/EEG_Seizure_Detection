"""PyTorch Dataset and DataLoader Utilities for Spatio-Temporal EEG Windows.

Wraps preprocessed windowed tensors (N, C, T) and binary seizure labels (N,)
with batch loading, class weight computation, and device management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGWindowDataset(Dataset):
    """PyTorch Dataset wrapping multichannel sliding EEG windows."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            windows: Array of shape (n_windows, n_channels, n_samples).
            labels: Array of shape (n_windows,) with binary labels {0, 1}.
        """
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        if len(self.windows) != len(self.labels):
            raise ValueError(
                f"Mismatch: windows count ({len(self.windows)}) != "
                f"labels count ({len(self.labels)})"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss weighting."""
        n_samples = len(self.labels)
        n_classes = 2
        counts = torch.bincount(self.labels, minlength=n_classes).float()
        # w_c = N / (n_classes * N_c)
        weights = n_samples / (n_classes * torch.clamp(counts, min=1.0))
        return weights


def load_npz_tensors(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Load windows and labels from a processed .npz archive."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed archive not found: {path}")

    data = np.load(path, allow_pickle=True)
    windows = data["windows"]  # (N, Channels, Samples)
    labels = data["labels"]    # (N,)
    meta = {
        "patient_id": str(data["patient_id"]),
        "recording_name": str(data["recording_name"]),
    }
    return windows, labels, meta


def create_eeg_dataloaders(
    train_npz: str | Path,
    val_npz: str | Path,
    test_npz: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train, validation, and test PyTorch DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
            - train_loader, val_loader, test_loader
            - class_weights tensor for CrossEntropyLoss
    """
    X_train, y_train, _ = load_npz_tensors(train_npz)
    X_val, y_val, _ = load_npz_tensors(val_npz)
    X_test, y_test, _ = load_npz_tensors(test_npz)

    train_ds = EEGWindowDataset(X_train, y_train)
    val_ds = EEGWindowDataset(X_val, y_val)
    test_ds = EEGWindowDataset(X_test, y_test)

    class_weights = train_ds.compute_class_weights()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_weights
