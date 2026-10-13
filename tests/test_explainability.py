"""Unit tests for NeuroGuard Explainability (XAI) Suite."""

import pytest
import numpy as np
import torch

from src.models.classical import XGBoostBaseline
from src.models.cnn1d import EEG1DCNN
from src.models.cnn_lstm import EEG_CNNLSTM
from src.explainability.explainer import EEGExplainer


@pytest.fixture
def dummy_classical_model():
    """Create a trained XGBoost model on dummy feature data."""
    X = np.random.randn(40, 10).astype(np.float32)
    y = np.array([0] * 30 + [1] * 10)
    feature_names = [f"ch{i}_alpha_power" for i in range(10)]
    model = XGBoostBaseline(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model, feature_names


@pytest.fixture
def dummy_cnn_model():
    """Create an initialized 1D CNN model."""
    model = EEG1DCNN(in_channels=4, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def dummy_cnn_lstm_model():
    """Create an initialized CNN-LSTM model."""
    model = EEG_CNNLSTM(in_channels=4, num_classes=2, lstm_hidden=16)
    model.eval()
    return model


def test_explainer_feature_importances(dummy_classical_model):
    """Test feature importance extraction and aggregation."""
    model, feature_names = dummy_classical_model
    explanations = EEGExplainer.get_feature_importances(
        model_wrapper=model,
        feature_names=feature_names,
        top_k=5,
    )

    assert "top_features" in explanations
    assert "channel_importance" in explanations
    assert "biomarker_importance" in explanations
    assert "disclaimer" in explanations

    # Verify top features count
    assert len(explanations["top_features"]) <= 5
    assert explanations["top_features"][0]["rank"] == 1

    # Verify disclaimer is non-empty
    assert "RESEARCH NOTICE" in explanations["disclaimer"]


def test_explainer_deep_learning_saliency(dummy_cnn_model, dummy_cnn_lstm_model):
    """Test gradient saliency calculation for 1D CNN and CNN-LSTM."""
    window = torch.randn(4, 256)

    # 1D CNN saliency
    sal_cnn = EEGExplainer.compute_deep_learning_saliency(
        model=dummy_cnn_model,
        window_tensor=window,
        target_class=1,
    )
    assert isinstance(sal_cnn, np.ndarray)
    assert sal_cnn.shape == (4, 256)
    assert np.max(sal_cnn) <= 1.0 + 1e-6
    assert np.min(sal_cnn) >= 0.0

    # CNN-LSTM saliency
    sal_lstm = EEGExplainer.compute_deep_learning_saliency(
        model=dummy_cnn_lstm_model,
        window_tensor=window,
        target_class=1,
    )
    assert isinstance(sal_lstm, np.ndarray)
    assert sal_lstm.shape == (4, 256)
    assert np.max(sal_lstm) <= 1.0 + 1e-6
    assert np.min(sal_lstm) >= 0.0
