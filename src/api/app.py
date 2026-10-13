"""NeuroGuard FastAPI Application — Real-Time Explainable EEG Seizure Detection."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config, get_project_root
from src.features.extractor import EEGFeatureExtractor
from src.models.classical import LogisticRegressionBaseline, RandomForestBaseline, XGBoostBaseline
from src.models.cnn1d import EEG1DCNN
from src.models.cnn_lstm import EEG_CNNLSTM
from src.explainability.explainer import EEGExplainer
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ModelPerformanceSummary,
    PredictFeaturesRequest,
    PredictWindowRequest,
    PredictionResponse,
    SinglePredictionResult,
    ExplainRequest,
    ExplainResponse,
)

# Initialize FastAPI App
app = FastAPI(
    title="NeuroGuard EEG Seizure Detection API",
    description=(
        "Production-grade REST API for patient-independent, real-time explainable "
        "epileptic seizure detection from 17-channel scalp EEG signals."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local dashboards and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLINICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE NOTICE: NeuroGuard is an academic machine learning decision-support "
    "prototype developed for the PhysioNet CHB-MIT Scalp EEG dataset. It is NOT FDA/CE approved "
    "for standalone clinical diagnosis or acute treatment decisions."
)

STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ"
]

# Model Cache
MODELS_CACHE: Dict[str, Any] = {}
FEATURE_NAMES: List[str] = []


def load_all_models() -> None:
    """Load all saved models into memory at startup."""
    global FEATURE_NAMES
    root = get_project_root()
    models_dir = root / "artifacts" / "models"
    processed_dir = root / "data" / "processed"

    # Feature column names
    FEATURE_NAMES = EEGFeatureExtractor.get_feature_names(STANDARD_CHANNELS)

    # 1. Classical Models
    xgboost_optuna_path = models_dir / "xgboost_optuna_best.joblib"
    if xgboost_optuna_path.exists():
        clf_opt = XGBoostBaseline()
        clf_opt.load(xgboost_optuna_path)
        MODELS_CACHE["xgboost_optuna"] = clf_opt

    xgboost_path = models_dir / "xgboost.joblib"
    if xgboost_path.exists():
        clf_xgb = XGBoostBaseline()
        clf_xgb.load(xgboost_path)
        MODELS_CACHE["xgboost"] = clf_xgb

    rf_path = models_dir / "random_forest.joblib"
    if rf_path.exists():
        clf_rf = RandomForestBaseline()
        clf_rf.load(rf_path)
        MODELS_CACHE["random_forest"] = clf_rf

    lr_path = models_dir / "logistic_regression.joblib"
    if lr_path.exists():
        clf_lr = LogisticRegressionBaseline()
        clf_lr.load(lr_path)
        MODELS_CACHE["logistic_regression"] = clf_lr

    # 2. Deep Learning Models
    cnn1d_path = models_dir / "cnn1d_best.pt"
    if cnn1d_path.exists():
        cnn1d = EEG1DCNN(in_channels=17, num_classes=2)
        cnn1d.load(cnn1d_path, map_location="cpu")
        MODELS_CACHE["cnn1d"] = cnn1d

    cnn_lstm_path = models_dir / "cnn_lstm_best.pt"
    if cnn_lstm_path.exists():
        cnn_lstm = EEG_CNNLSTM(in_channels=17, num_classes=2, lstm_hidden=64)
        cnn_lstm.load(cnn_lstm_path, map_location="cpu")
        MODELS_CACHE["cnn_lstm"] = cnn_lstm


# Load models on module import
load_all_models()


@app.get("/", tags=["System"])
def root_endpoint() -> Dict[str, str]:
    """Root health and welcome endpoint."""
    return {
        "project": "NeuroGuard EEG Seizure Detection",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "disclaimer": CLINICAL_DISCLAIMER,
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return system readiness, device, and active model inventory."""
    return HealthResponse(
        status="healthy" if len(MODELS_CACHE) > 0 else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=list(MODELS_CACHE.keys()),
        device="cpu",
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info() -> ModelInfoResponse:
    """Return input specifications, supported channels, and test benchmark metrics."""
    benchmarks = [
        ModelPerformanceSummary(
            model_name="XGBoost (Optuna Tuned)",
            accuracy=0.8638,
            sensitivity_recall=1.0000,
            specificity=0.8555,
            precision=0.2967,
            f1_score=0.4576,
            roc_auc=0.9659,
            pr_auc=0.5268,
        ),
        ModelPerformanceSummary(
            model_name="CNN-LSTM Spatio-Temporal",
            accuracy=0.8511,
            sensitivity_recall=0.8148,
            specificity=0.8533,
            precision=0.2529,
            f1_score=0.3860,
            roc_auc=0.8379,
            pr_auc=0.7751,
        ),
        ModelPerformanceSummary(
            model_name="1D CNN Baseline",
            accuracy=0.6404,
            sensitivity_recall=0.8519,
            specificity=0.6275,
            precision=0.1223,
            f1_score=0.2140,
            roc_auc=0.8111,
            pr_auc=0.2450,
        ),
        ModelPerformanceSummary(
            model_name="Random Forest Baseline",
            accuracy=0.9383,
            sensitivity_recall=0.2222,
            specificity=0.9819,
            precision=0.4286,
            f1_score=0.2927,
            roc_auc=0.8906,
            pr_auc=0.4286,
        ),
        ModelPerformanceSummary(
            model_name="Logistic Regression Baseline",
            accuracy=0.8660,
            sensitivity_recall=0.4815,
            specificity=0.8894,
            precision=0.2097,
            f1_score=0.2921,
            roc_auc=0.7410,
            pr_auc=0.2097,
        ),
    ]

    return ModelInfoResponse(
        project_name="NeuroGuard",
        description="Patient-Independent Explainable EEG Seizure Detection on CHB-MIT Scalp Database",
        sampling_rate_hz=256,
        window_duration_seconds=4.0,
        window_samples=1024,
        num_channels=17,
        channel_names=STANDARD_CHANNELS,
        available_models=list(MODELS_CACHE.keys()),
        default_model="xgboost_optuna" if "xgboost_optuna" in MODELS_CACHE else "xgboost",
        benchmarks=benchmarks,
        disclaimer=CLINICAL_DISCLAIMER,
    )


@app.post("/predict/features", response_model=PredictionResponse, tags=["Inference"])
def predict_from_features(request: PredictFeaturesRequest) -> PredictionResponse:
    """Predict seizure occurrence from 306-dimensional biomarker feature vectors."""
    t0 = time.perf_counter()

    model_key = request.model_name or "xgboost_optuna"
    if model_key not in MODELS_CACHE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_key}' is not available. Available models: {list(MODELS_CACHE.keys())}",
        )

    model = MODELS_CACHE[model_key]
    if isinstance(model, (torch.nn.Module,)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' is a deep learning model. Use `/predict/raw-window` instead.",
        )

    feats = np.array(request.features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    if feats.shape[1] != 306:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature matrix must have 306 columns, but received {feats.shape[1]}.",
        )

    probs = model.predict_proba(feats)
    preds = model.predict(feats)

    results = []
    for idx, (p, cls_pred) in enumerate(zip(probs, preds)):
        seizure_prob = float(p)
        is_seizure = bool(cls_pred == 1)
        conf = seizure_prob if is_seizure else (1.0 - seizure_prob)
        results.append(
            SinglePredictionResult(
                segment_index=idx,
                seizure_detected=is_seizure,
                seizure_probability=round(seizure_prob, 4),
                label="SEIZURE (Ictal)" if is_seizure else "NON-SEIZURE (Interictal)",
                confidence=round(conf, 4),
            )
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    seizure_count = sum(1 for r in results if r.seizure_detected)

    return PredictionResponse(
        model_used=model_key,
        predictions=results,
        summary={
            "total_segments": len(results),
            "seizure_segments_detected": seizure_count,
            "alarm_active": seizure_count > 0,
        },
        latency_ms=round(latency_ms, 2),
        disclaimer=CLINICAL_DISCLAIMER,
    )


@app.post("/predict/raw-window", response_model=PredictionResponse, tags=["Inference"])
def predict_raw_window(request: PredictWindowRequest) -> PredictionResponse:
    """Predict seizure probability directly from a 17-channel x 1024-sample EEG voltage segment."""
    t0 = time.perf_counter()

    model_key = request.model_name or "cnn_lstm"
    if model_key not in MODELS_CACHE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_key}' not found. Available models: {list(MODELS_CACHE.keys())}",
        )

    model = MODELS_CACHE[model_key]
    if not isinstance(model, torch.nn.Module):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' is a classical feature model. Use `/predict/features` instead.",
        )

    tensor_in = np.array(request.window, dtype=np.float32)
    if tensor_in.shape != (17, 1024):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Raw EEG window must have shape (17 channels, 1024 samples), got {tensor_in.shape}.",
        )

    # Normalize window per-channel zero mean, unit variance
    means = np.mean(tensor_in, axis=1, keepdims=True)
    stds = np.std(tensor_in, axis=1, keepdims=True) + 1e-6
    norm_tensor = (tensor_in - means) / stds

    x = torch.from_numpy(norm_tensor).unsqueeze(0)  # Shape: (1, 17, 1024)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    seizure_prob = float(probs[1])
    is_seizure = seizure_prob >= 0.5
    conf = seizure_prob if is_seizure else (1.0 - seizure_prob)

    single_result = SinglePredictionResult(
        segment_index=0,
        seizure_detected=is_seizure,
        seizure_probability=round(seizure_prob, 4),
        label="SEIZURE (Ictal)" if is_seizure else "NON-SEIZURE (Interictal)",
        confidence=round(conf, 4),
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        model_used=model_key,
        predictions=[single_result],
        summary={
            "total_segments": 1,
            "seizure_segments_detected": 1 if is_seizure else 0,
            "alarm_active": is_seizure,
        },
        latency_ms=round(latency_ms, 2),
        disclaimer=CLINICAL_DISCLAIMER,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_model_attributions(request: ExplainRequest) -> ExplainResponse:
    """Return top biomarker and channel-level attributions for tree-based models."""
    model_key = request.model_name or "xgboost_optuna"
    if model_key not in MODELS_CACHE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_key}' not found.",
        )

    model = MODELS_CACHE[model_key]
    if isinstance(model, torch.nn.Module):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' is a deep learning model. Global feature attribution is for tree models.",
        )

    explanations = EEGExplainer.get_feature_importances(
        model_wrapper=model,
        feature_names=FEATURE_NAMES,
        top_k=request.top_k,
    )

    return ExplainResponse(
        model_used=model_key,
        top_features=explanations["top_features"],
        channel_importance=explanations["channel_importance"],
        biomarker_importance=explanations["biomarker_importance"],
        disclaimer=CLINICAL_DISCLAIMER,
    )
