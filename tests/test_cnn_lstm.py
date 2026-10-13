"""Unit tests for spatio-temporal CNN-LSTM architecture."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.cnn_lstm import EEG_CNNLSTM


def test_cnn_lstm_forward_pass_and_shapes():
    """Verify CNN-LSTM forward pass logits and probability dimensions."""
    batch_size = 4
    in_channels = 17
    seq_len = 1024

    model = EEG_CNNLSTM(in_channels=in_channels, num_classes=2, lstm_hidden=32)
    x = torch.randn(batch_size, in_channels, seq_len)

    logits = model(x)
    assert logits.shape == (batch_size, 2)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert model.get_num_parameters() > 5000


def test_cnn_lstm_save_and_load(tmp_path: Path):
    """Verify CNN-LSTM model serialization and fidelity."""
    model = EEG_CNNLSTM(in_channels=17, num_classes=2, lstm_hidden=32)
    ckpt_path = tmp_path / "test_cnn_lstm.pt"

    model.save(ckpt_path)
    assert ckpt_path.exists()

    loaded = EEG_CNNLSTM(in_channels=17, num_classes=2, lstm_hidden=32)
    loaded.load(ckpt_path)

    model.eval()
    x = torch.randn(2, 17, 1024)
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded(x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5)
