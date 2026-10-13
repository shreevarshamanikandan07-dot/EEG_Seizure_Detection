"""Training Pipeline for 1D CNN Multichannel EEG Seizure Classifier.

Trains EEG1DCNN with weighted cross-entropy loss, ReduceLROnPlateau,
early stopping, and evaluates on held-out patient chb03.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.utils.seed import set_seed
from src.data.torch_dataset import create_eeg_dataloaders
from src.models.cnn1d import EEG1DCNN
from src.evaluation.metrics import calculate_metrics, save_metrics_json


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run model inference over DataLoader and return true, predicted labels, and loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return (
        np.array(all_y_true, dtype=int),
        np.array(all_y_pred, dtype=int),
        np.array(all_y_prob, dtype=float),
        avg_loss,
    )


def train_cnn(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 4,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Train 1D CNN on patient chb01, validate on chb02, and test on chb03."""
    set_seed(42)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 75)
    print(" NeuroGuard: Deep Learning 1D CNN Training & Evaluation")
    print(f" Device: {device}")
    print("=" * 75)

    train_npz = processed_dir / "chb01_chb01_03.npz"
    val_npz = processed_dir / "chb02_chb02_16.npz"
    test_npz = processed_dir / "chb03_chb03_01.npz"

    train_loader, val_loader, test_loader, class_weights = create_eeg_dataloaders(
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        batch_size=batch_size,
    )

    class_weights = class_weights.to(device)
    print(f"[*] Computed Class Loss Weights: Non-Seizure={class_weights[0]:.2f}, Seizure={class_weights[1]:.2f}")

    # Inspect input channel dimension
    sample_x, _ = next(iter(train_loader))
    in_channels = sample_x.shape[1]
    seq_len = sample_x.shape[2]
    print(f"[*] Input Tensor Dimensions: Channels={in_channels}, Sequence Length={seq_len}")

    model = EEG1DCNN(in_channels=in_channels, num_classes=2).to(device)
    print(f"[*] Total Trainable Parameters: {model.get_num_parameters():,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = models_dir / "cnn1d_best.pt"

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_roc_auc": []}

    print("-" * 75)
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val F1':<8} | {'Val ROC-AUC':<11} | {'Status':<10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(train_loader.dataset)

        # Evaluate on validation set
        y_val_true, y_val_pred, y_val_prob, val_loss = evaluate_model_on_loader(model, val_loader, device)
        val_m = calculate_metrics(y_val_true, y_val_pred, y_val_prob)

        scheduler.step(val_m.f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_m.f1)
        history["val_roc_auc"].append(val_m.roc_auc)

        status_str = ""
        if val_m.f1 > best_val_f1:
            best_val_f1 = val_m.f1
            best_epoch = epoch
            epochs_no_improve = 0
            model.save(best_model_path)
            status_str = f"[*] Best (ckpt saved)"
        else:
            epochs_no_improve += 1
            status_str = f"Patience: {epochs_no_improve}/{patience}"

        print(
            f"{epoch:<7} | {train_loss:<10.4f} | {val_loss:<10.4f} | "
            f"{val_m.f1:<8.4f} | {val_m.roc_auc:<11.4f} | {status_str}"
        )

        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Load best checkpoint for final evaluation
    print(f"\n[*] Reloading best model from {best_model_path} (epoch {best_epoch})...")
    model.load(best_model_path, map_location=str(device))

    # Evaluate on Validation
    y_val_t, y_val_p, y_val_prob, _ = evaluate_model_on_loader(model, val_loader, device)
    final_val_m = calculate_metrics(y_val_t, y_val_p, y_val_prob)

    # Evaluate on Held-Out Test
    y_test_t, y_test_p, y_test_prob, _ = evaluate_model_on_loader(model, test_loader, device)
    final_test_m = calculate_metrics(y_test_t, y_test_p, y_test_prob)

    print("\n" + "=" * 75)
    print(" 1D CNN Benchmark Evaluation Results")
    print("=" * 75)
    print(f"Validation Set (Patient chb02):")
    print(final_val_m.summary_table())
    print("-" * 75)
    print(f"Held-Out Test Set (Patient chb03):")
    print(final_test_m.summary_table())
    print("=" * 75)

    benchmark_data = {
        "model": "EEG1DCNN",
        "best_epoch": best_epoch,
        "parameters": model.get_num_parameters(),
        "checkpoint": str(best_model_path),
        "validation_metrics": final_val_m.to_dict(),
        "test_metrics": final_test_m.to_dict(),
        "training_history": history,
    }

    metrics_file = metrics_dir / "cnn1d_benchmark.json"
    save_metrics_json(benchmark_data, metrics_file)
    print(f"[OK] Deep Learning benchmark saved to: {metrics_file}")
    return benchmark_data


if __name__ == "__main__":
    train_cnn()
