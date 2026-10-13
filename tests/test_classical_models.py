"""Unit tests for classical machine-learning models and evaluation metrics."""

import pytest
import numpy as np
from pathlib import Path

from src.models.classical import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
    get_classical_model,
)
from src.evaluation.metrics import calculate_metrics


@pytest.fixture
def synthetic_features():
    """Generate synthetic separable feature matrix for 2-class problem."""
    np.random.seed(42)
    n_samples = 80
    n_features = 20

    # Class 0: centered at -1
    X_0 = np.random.randn(n_samples // 2, n_features) - 1.0
    y_0 = np.zeros(n_samples // 2, dtype=int)

    # Class 1: centered at +1
    X_1 = np.random.randn(n_samples // 2, n_features) + 1.0
    y_1 = np.ones(n_samples // 2, dtype=int)

    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])
    return X, y


def test_logistic_regression_fit_and_predict(synthetic_features, tmp_path: Path):
    """Test LogisticRegressionBaseline training, inference, and serialization."""
    X, y = synthetic_features
    clf = LogisticRegressionBaseline(C=1.0, max_iter=200)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == len(y)
    assert len(probs) == len(y)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Accuracy on linearly separated data should be high (>85%)
    assert np.mean(preds == y) > 0.85

    # Test serialization
    ckpt = tmp_path / "lr.joblib"
    clf.save(ckpt)
    assert ckpt.exists()

    loaded = LogisticRegressionBaseline()
    loaded.load(ckpt)
    assert loaded.is_trained
    np.testing.assert_allclose(loaded.predict_proba(X), probs)


def test_random_forest_fit_and_predict(synthetic_features):
    """Test RandomForestBaseline training and probability outputs."""
    X, y = synthetic_features
    clf = RandomForestBaseline(n_estimators=20, max_depth=5)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == len(y)
    assert len(probs) == len(y)
    assert np.mean(preds == y) > 0.85


def test_xgboost_fit_and_predict(synthetic_features):
    """Test XGBoostBaseline training and probability outputs."""
    X, y = synthetic_features
    clf = XGBoostBaseline(n_estimators=20, max_depth=3)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == len(y)
    assert len(probs) == len(y)
    assert np.mean(preds == y) > 0.85


def test_metrics_calculation():
    """Verify diagnostic metric calculation (accuracy, sensitivity, specificity, F1)."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9, 0.8, 0.85, 0.4])

    m = calculate_metrics(y_true, y_pred, y_prob)

    # 3 TN, 1 FP, 1 FN, 3 TP
    assert m.true_negatives == 3
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.true_positives == 3
    assert m.accuracy == 6 / 8
    assert m.recall == 3 / 4       # Sensitivity = 0.75
    assert m.specificity == 3 / 4  # Specificity = 0.75
    assert m.roc_auc > 0.7
