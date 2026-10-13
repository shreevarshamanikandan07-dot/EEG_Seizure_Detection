"""Classical Machine Learning Model Baselines for EEG Seizure Detection.

Provides unified wrappers for:
1. Regularized Logistic Regression with balanced weighting.
2. Random Forest with bootstrap ensemble and class balancing.
3. Gradient Boosted Decision Trees (XGBoost) with scale_pos_weight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class BaseClassicalModel(ABC):
    """Abstract base class for classical EEG seizure classifiers."""

    def __init__(self, name: str, random_state: int = 42) -> None:
        self.name = name
        self.random_state = random_state
        self.model: Any = None
        self.is_trained: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> BaseClassicalModel:
        """Fit model to training feature matrix."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary seizure labels (0 or 1)."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.name} is not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of seizure class (label 1)."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.name} is not trained.")
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 2:
            return probas[:, 1]
        return probas[:, 0]

    def save(self, filepath: str | Path) -> None:
        """Save fitted model checkpoint to disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "model": self.model, "random_state": self.random_state}, target)

    def load(self, filepath: str | Path) -> BaseClassicalModel:
        """Load fitted model checkpoint from disk."""
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {target}")
        data = joblib.load(target)
        self.name = data["name"]
        self.model = data["model"]
        self.random_state = data.get("random_state", 42)
        self.is_trained = True
        return self


class LogisticRegressionBaseline(BaseClassicalModel):
    """L2-regularized Logistic Regression with balanced weighting and feature scaling."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        super().__init__(name="Logistic_Regression", random_state=random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LogisticRegressionBaseline:
        # Preprocessing: standard scaler pipeline handles heterogeneous biomarker scales
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )),
        ])
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self


class RandomForestBaseline(BaseClassicalModel):
    """Random Forest Classifier with balanced bootstrap weighting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="Random_Forest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> RandomForestBaseline:
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )),
        ])
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self


class XGBoostBaseline(BaseClassicalModel):
    """Gradient Boosted Decision Trees with scale_pos_weight for class imbalance."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="XGBoost", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> XGBoostBaseline:
        # Calculate scale_pos_weight automatically if not provided (ratio of negatives to positives)
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        spw = self.scale_pos_weight if self.scale_pos_weight is not None else (n_neg / max(1, n_pos))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", xgb_clf),
        ])

        if eval_set is not None:
            # Fit scaler first to transform eval set consistently
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            xgb_clf.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            self.model = Pipeline([("scaler", scaler), ("classifier", xgb_clf)])
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True
        return self


def get_classical_model(model_type: str, **kwargs: Any) -> BaseClassicalModel:
    """Factory method for classical models."""
    norm_type = model_type.lower().replace("-", "_").replace(" ", "_")
    if norm_type in ["logistic_regression", "lr"]:
        return LogisticRegressionBaseline(**kwargs)
    elif norm_type in ["random_forest", "rf"]:
        return RandomForestBaseline(**kwargs)
    elif norm_type in ["xgboost", "xgb"]:
        return XGBoostBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")
