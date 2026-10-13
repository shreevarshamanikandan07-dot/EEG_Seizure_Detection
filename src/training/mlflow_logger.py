"""MLflow-Compatible Experiment Tracking System for NeuroGuard.

Provides structured logging of hyperparameters, validation metrics, trial durations,
and model artifacts to artifacts/mlruns/ using standard MLflow schema.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils.config import get_project_root


@dataclass
class RunRecord:
    run_id: str
    run_name: str
    experiment_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "RUNNING"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class MLflowTracker:
    """Standardized Experiment Logger maintaining artifacts/mlruns directory store."""

    def __init__(
        self,
        experiment_name: str = "NeuroGuard_EEG_Seizure_Detection",
        mlruns_dir: Optional[str | Path] = None,
    ) -> None:
        self.experiment_name = experiment_name
        root = get_project_root()
        self.mlruns_dir = Path(mlruns_dir) if mlruns_dir else root / "artifacts" / "mlruns"
        self.mlruns_dir.mkdir(parents=True, exist_ok=True)

        self.current_run: Optional[RunRecord] = None
        self._init_experiment_meta()

    def _init_experiment_meta(self) -> None:
        """Create MLflow meta.yaml if not present."""
        exp_meta_path = self.mlruns_dir / "meta.yaml"
        if not exp_meta_path.exists():
            meta = {
                "artifact_location": str(self.mlruns_dir.resolve()),
                "creation_time": int(time.time() * 1000),
                "experiment_id": "0",
                "last_update_time": int(time.time() * 1000),
                "lifecycle_stage": "active",
                "name": self.experiment_name,
            }
            with open(exp_meta_path, "w", encoding="utf-8") as f:
                yaml.dump(meta, f)

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run."""
        run_id = uuid.uuid4().hex[:12]
        self.current_run = RunRecord(
            run_id=run_id,
            run_name=run_name,
            experiment_name=self.experiment_name,
            start_time=time.time(),
            tags=tags or {},
        )
        run_dir = self.mlruns_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "params").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        return run_id

    def log_param(self, key: str, value: Any) -> None:
        """Log hyperparameter."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self.current_run.params[key] = value
        run_dir = self.mlruns_dir / self.current_run.run_id
        param_file = run_dir / "params" / key
        param_file.write_text(str(value), encoding="utf-8")

    def log_params(self, params_dict: Dict[str, Any]) -> None:
        """Log dictionary of hyperparameters."""
        for k, v in params_dict.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float) -> None:
        """Log evaluated metric."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        val_float = float(value)
        self.current_run.metrics[key] = val_float
        run_dir = self.mlruns_dir / self.current_run.run_id
        metric_file = run_dir / "metrics" / key
        timestamp = int(time.time() * 1000)
        with open(metric_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} {val_float} 0\n")

    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """Log dictionary of metrics."""
        for k, v in metrics_dict.items():
            self.log_metric(k, v)

    def log_artifact(self, filepath: str | Path) -> None:
        """Record model artifact associated with run."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        src = Path(filepath)
        if src.exists():
            dest = self.mlruns_dir / self.current_run.run_id / "artifacts" / src.name
            import shutil
            shutil.copy2(src, dest)
            self.current_run.artifacts.append(str(dest))

    def end_run(self, status: str = "FINISHED") -> RunRecord:
        """Conclude active run and persist summary."""
        if self.current_run is None:
            raise RuntimeError("No active run to end.")
        self.current_run.end_time = time.time()
        self.current_run.status = status

        run_dir = self.mlruns_dir / self.current_run.run_id
        meta_run = {
            "run_id": self.current_run.run_id,
            "run_name": self.current_run.run_name,
            "experiment_name": self.current_run.experiment_name,
            "status": self.current_run.status,
            "start_time": int(self.current_run.start_time * 1000),
            "end_time": int(self.current_run.end_time * 1000),
            "duration_sec": round(self.current_run.end_time - self.current_run.start_time, 2),
            "params": self.current_run.params,
            "metrics": self.current_run.metrics,
            "tags": self.current_run.tags,
        }
        with open(run_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta_run, f)

        # Append to overall summary JSON
        summary_path = self.mlruns_dir / "runs_summary.json"
        summary_list = []
        if summary_path.exists():
            try:
                summary_list = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_list = []
        summary_list.append(meta_run)
        summary_path.write_text(json.dumps(summary_list, indent=2), encoding="utf-8")

        completed = self.current_run
        self.current_run = None
        return completed
