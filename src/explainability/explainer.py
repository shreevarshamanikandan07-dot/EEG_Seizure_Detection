"""Explainable Artificial Intelligence (XAI) for EEG Seizure Detection.

Provides:
1. Feature & channel-level importance analysis for classical models (XGBoost, Random Forest).
2. Physiological frequency band attribution breakdown.
3. Gradient-based input saliency attribution for deep-learning models (1D CNN, CNN-LSTM).
4. Academic research caveats distinguishing correlation from clinical causation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class EEGExplainer:
    """XAI suite providing interpretable physiological and neural attributions."""

    CLINICAL_DISCLAIMER = (
        "RESEARCH NOTICE: Model attribution values reflect algorithmic pattern sensitivity "
        "and correlation with annotated training data. They do NOT establish clinical etiology "
        "or imply that specific channels or frequency bands physiologically cause seizure genesis."
    )

    @classmethod
    def get_feature_importances(
        cls,
        model_wrapper: Any,
        feature_names: List[str],
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Extract top feature importances from a tree-based classifier (XGBoost / Random Forest).

        Args:
            model_wrapper: Trained instance of XGBoostBaseline or RandomForestBaseline.
            feature_names: List of all 306 feature column names.
            top_k: Number of highest-ranked features to return.

        Returns:
            Dict containing top features, channel aggregates, and band aggregates.
        """
        # Retrieve underlying scikit-learn or XGBoost classifier
        pipeline = model_wrapper.model
        clf = pipeline.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            raw_importances = clf.feature_importances_
        else:
            raise ValueError("Provided model does not expose feature_importances_ attribute.")

        # Normalize to sum to 1.0
        norm_imp = raw_importances / (np.sum(raw_importances) + 1e-12)

        # Sort descending
        sorted_indices = np.argsort(norm_imp)[::-1]
        top_features = []
        for idx in sorted_indices[:top_k]:
            top_features.append({
                "feature": feature_names[idx],
                "importance": float(norm_imp[idx]),
                "rank": len(top_features) + 1,
            })

        # Channel-level aggregated importance
        channel_importance: Dict[str, float] = {}
        # Band/Biomarker aggregated importance
        biomarker_importance: Dict[str, float] = {}

        for feat_name, imp_val in zip(feature_names, norm_imp):
            # Format: 'FP1-F7_alpha_power' -> channel = 'FP1-F7', biomarker = 'alpha_power'
            parts = feat_name.split("_", 1)
            ch = parts[0]
            biomarker = parts[1] if len(parts) > 1 else "other"

            channel_importance[ch] = channel_importance.get(ch, 0.0) + float(imp_val)
            biomarker_importance[biomarker] = biomarker_importance.get(biomarker, 0.0) + float(imp_val)

        # Sort dictionaries descending
        sorted_channels = dict(sorted(channel_importance.items(), key=lambda x: x[1], reverse=True))
        sorted_biomarkers = dict(sorted(biomarker_importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "top_features": top_features,
            "channel_importance": sorted_channels,
            "biomarker_importance": sorted_biomarkers,
            "disclaimer": cls.CLINICAL_DISCLAIMER,
        }

    @classmethod
    def compute_deep_learning_saliency(
        cls,
        model: nn.Module,
        window_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute Gradient x Input temporal saliency map for a single EEG window.

        Args:
            model: PyTorch model (EEG1DCNN or EEG_CNNLSTM).
            window_tensor: Input tensor of shape (1, channels, seq_len) or (channels, seq_len).
            target_class: Class index to explain (default 1 for seizure).

        Returns:
            np.ndarray: Saliency attribution array of shape (channels, seq_len).
        """
        model.eval()
        if window_tensor.ndim == 2:
            x = window_tensor.unsqueeze(0).clone().detach()
        else:
            x = window_tensor.clone().detach()

        x.requires_grad = True

        logits = model(x)
        score = logits[0, target_class]
        score.backward()

        # Input x Gradient attribution
        grads = x.grad[0].detach().cpu().numpy()
        inputs = x[0].detach().cpu().numpy()
        saliency = np.abs(inputs * grads)

        # Normalize per channel
        max_val = np.max(saliency) + 1e-12
        return saliency / max_val
