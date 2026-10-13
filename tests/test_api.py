"""Unit and integration tests for NeuroGuard FastAPI REST API."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root welcome endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "NeuroGuard EEG Seizure Detection"
    assert data["status"] == "operational"
    assert "disclaimer" in data


def test_health_check_endpoint():
    """Test the health check endpoint returns loaded models and cpu device."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "models_loaded" in data
    assert len(data["models_loaded"]) > 0
    assert data["device"] == "cpu"


def test_model_info_endpoint():
    """Test model metadata, input specifications, and benchmarks."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["sampling_rate_hz"] == 256
    assert data["num_channels"] == 17
    assert len(data["channel_names"]) == 17
    assert len(data["benchmarks"]) >= 3
    assert "disclaimer" in data


def test_predict_features_success():
    """Test prediction from a 306-dimensional biomarker feature vector."""
    dummy_features = [np.random.randn(306).tolist()]
    payload = {
        "features": dummy_features,
        "model_name": "xgboost_optuna" if "xgboost_optuna" in response_models() else "xgboost",
    }
    response = client.post("/predict/features", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert "seizure_detected" in data["predictions"][0]
    assert "seizure_probability" in data["predictions"][0]
    assert 0.0 <= data["predictions"][0]["seizure_probability"] <= 1.0
    assert "disclaimer" in data
    assert "latency_ms" in data


def test_predict_features_validation_error():
    """Test that wrong feature dimensions trigger 422 Unprocessable Entity."""
    bad_features = [[1.0, 2.0, 3.0]]  # Only 3 features instead of 306
    payload = {
        "features": bad_features,
        "model_name": "xgboost",
    }
    response = client.post("/predict/features", json=payload)
    assert response.status_code == 422


def test_predict_raw_window_success():
    """Test prediction from a raw (17, 1024) EEG window using deep learning."""
    raw_window = np.random.randn(17, 1024).tolist()
    payload = {
        "window": raw_window,
        "model_name": "cnn_lstm",
    }
    response = client.post("/predict/raw-window", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert pred["seizure_detected"] in (True, False)
    assert 0.0 <= pred["seizure_probability"] <= 1.0
    assert "disclaimer" in data


def test_predict_raw_window_validation_error():
    """Test that wrong window shape triggers 422 Unprocessable Entity."""
    bad_window = np.random.randn(8, 512).tolist()
    payload = {
        "window": bad_window,
        "model_name": "cnn_lstm",
    }
    response = client.post("/predict/raw-window", json=payload)
    assert response.status_code == 422


def test_explain_endpoint():
    """Test explanation attributions for tree-based model."""
    payload = {
        "model_name": "xgboost_optuna" if "xgboost_optuna" in response_models() else "xgboost",
        "top_k": 10,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) <= 10
    assert "channel_importance" in data
    assert "biomarker_importance" in data
    assert "disclaimer" in data


def response_models():
    """Helper to get available models."""
    res = client.get("/health").json()
    return res.get("models_loaded", [])
