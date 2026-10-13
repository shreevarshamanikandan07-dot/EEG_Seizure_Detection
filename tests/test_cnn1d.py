"""Unit tests for 1D CNN architecture and PyTorch dataset."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.cnn1d import EEG1DCNN
from src.data.torch_dataset import EEGWindowDataset


def test_cnn1d_forward_pass_and_shapes():
    """Verify 1D CNN forward pass and probability output dimensions."""
    batch_size = 4
    in_channels = 17
    seq_len = 1024

    model = EEG1DCNN(in_channels=in_channels, num_classes=2)
    x = torch.randn(batch_size, in_channels, seq_len)

    logits = model(x)
    assert logits.shape == (batch_size, 2)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert model.get_num_parameters() > 10000


def test_cnn1d_save_and_load(tmp_path: Path):
    """Verify model weight saving and loading without loss of fidelity."""
    model = EEG1DCNN(in_channels=17, num_classes=2)
    ckpt_path = tmp_path / "test_cnn.pt"

    model.save(ckpt_path)
    assert ckpt_path.exists()

    loaded = EEG1DCNN(in_channels=17, num_classes=2)
    loaded.load(ckpt_path)

    model.eval()
    x = torch.randn(2, 17, 1024)
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded(x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5)


def test_eeg_window_dataset():
    """Verify EEGWindowDataset tensor casting, length, and class weights."""
    n_windows = 20
    n_channels = 17
    n_samples = 1024

    windows = np.random.randn(n_windows, n_channels, n_samples).astype(np.float32)
    # 18 non-seizures, 2 seizures
    labels = np.array([0] * 18 + [1] * 2, dtype=np.int64)

    dataset = EEGWindowDataset(windows, labels)
    assert len(dataset) == n_windows

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (n_channels, n_samples)
    assert y.item() == 0

    weights = dataset.compute_class_weights()
    assert len(weights) == 2
    # Rare class 1 should have much higher loss weight than majority class 0
    assert weights[1] > weights[0]
