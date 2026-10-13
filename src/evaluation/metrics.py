"""Comprehensive Evaluation Metrics for Biomedical Seizure Detection.

Calculates Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC,
PR-AUC (Average Precision), and full Confusion Matrix statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationMetrics:
    """Stores diagnostic metrics for clinical seizure detection."""
    accuracy: float
    precision: float
    recall: float          # Sensitivity
    specificity: float     # True Negative Rate
    f1: float
    roc_auc: float
    pr_auc: float          # Average Precision
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
    total_seizures: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_table(self) -> str:
        """Return formatted ASCII table of metrics."""
        lines = [
            f"  Accuracy:         {self.accuracy:.4f} ({self.accuracy*100:.2f}%)",
            f"  Sensitivity/Rec:  {self.recall:.4f} ({self.recall*100:.2f}%)",
            f"  Specificity:      {self.specificity:.4f} ({self.specificity*100:.2f}%)",
            f"  Precision:        {self.precision:.4f} ({self.precision*100:.2f}%)",
            f"  F1-Score:         {self.f1:.4f}",
            f"  ROC-AUC:          {self.roc_auc:.4f}",
            f"  PR-AUC:           {self.pr_auc:.4f}",
            f"  Confusion Matrix: TP={self.true_positives}, FP={self.false_positives}, FN={self.false_negatives}, TN={self.true_negatives}",
        ]
        return "\n".join(lines)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Calculate clinical seizure detection metrics.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1) or None if deriving from y_prob.
        y_prob: Predicted probability for positive class (seizure).
        threshold: Decision threshold if converting y_prob to binary predictions.

    Returns:
        EvaluationMetrics: Dataclass of evaluated clinical metrics.
    """
    y_true = np.asarray(y_true).astype(int)

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
    elif y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
    else:
        raise ValueError("Either y_pred or y_prob must be provided.")

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC and PR-AUC require probabilities
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(np.mean(y_true))
    else:
        roc_auc = float(acc)
        pr_auc = float(prec)

    return EvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_samples=len(y_true),
        total_seizures=int(np.sum(y_true)),
    )


def save_metrics_json(metrics_dict: Dict[str, Any], filepath: str | Path) -> None:
    """Save metrics dictionary to formatted JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
