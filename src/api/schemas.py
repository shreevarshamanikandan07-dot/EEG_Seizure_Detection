"""Pydantic schemas for NeuroGuard FastAPI REST service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check and loaded model inventory."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    timestamp: str
    models_loaded: List[str]
    device: str
    version: str = "1.0.0"


class ModelPerformanceSummary(BaseModel):
    """Patient-independent test metrics on held-out subject chb03."""
    model_name: str
    accuracy: float
    sensitivity_recall: float
    specificity: float
    precision: float
    f1_score: float
    roc_auc: float
    pr_auc: float


class ModelInfoResponse(BaseModel):
    """Model catalog, architecture metadata, and benchmark results."""
    project_name: str = "NeuroGuard"
    description: str
    sampling_rate_hz: int = 256
    window_duration_seconds: float = 4.0
    window_samples: int = 1024
    num_channels: int = 17
    channel_names: List[str]
    available_models: List[str]
    default_model: str
    benchmarks: List[ModelPerformanceSummary]
    disclaimer: str


class PredictFeaturesRequest(BaseModel):
    """Inference request using pre-extracted 306-biomarker feature vector(s)."""
    features: List[List[float]] = Field(
        ...,
        description="2D array of shape (N, 306) containing extracted EEG biomarkers."
    )
    model_name: Optional[str] = Field(
        default="xgboost_optuna",
        description="Model identifier: 'xgboost_optuna', 'xgboost', 'random_forest', 'logistic_regression'."
    )


class PredictWindowRequest(BaseModel):
    """Inference request using raw multi-channel EEG window."""
    window: List[List[float]] = Field(
        ...,
        description="2D array of shape (17, 1024) containing raw EEG voltage timeseries."
    )
    model_name: Optional[str] = Field(
        default="cnn_lstm",
        description="Deep learning model identifier: 'cnn_lstm' or 'cnn1d'."
    )


class SinglePredictionResult(BaseModel):
    """Prediction result for a single EEG window segment."""
    segment_index: int
    seizure_detected: bool
    seizure_probability: float
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response containing prediction outcomes, execution latency, and medical disclaimer."""
    model_used: str
    predictions: List[SinglePredictionResult]
    summary: Dict[str, Any]
    latency_ms: float
    disclaimer: str


class ExplainRequest(BaseModel):
    """Request to obtain top biomarker and channel feature attributions."""
    model_name: Optional[str] = Field(default="xgboost_optuna", description="Model name to explain.")
    top_k: int = Field(default=15, ge=1, le=50, description="Number of top features to return.")


class ExplainResponse(BaseModel):
    """Explainability attributions breakdown."""
    model_used: str
    top_features: List[Dict[str, Any]]
    channel_importance: Dict[str, float]
    biomarker_importance: Dict[str, float]
    disclaimer: str
