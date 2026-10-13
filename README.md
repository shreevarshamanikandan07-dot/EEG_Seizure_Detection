# 🧠 NeuroGuard: Patient-Independent Explainable EEG Seizure Detection Using Deep Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)](https://pytorch.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Academic Research & Decision-Support Prototype Notice:**
> NeuroGuard is an advanced biomedical machine learning research system developed as an academic CAT II project. It is **NOT** a clinically validated medical diagnostic device and must not replace professional clinical evaluation by qualified neurologists.

---

## 📌 1. Project Overview & Research Objectives

Epilepsy affects over 50 million individuals globally. Automated scalp EEG seizure detection is notoriously challenging due to:
1. **Severe Class Imbalance**: Seizure events typically constitute less than 1% of continuous multichannel EEG telemetry.
2. **Subject Heterogeneity**: Patient-specific morphological variance leads to severe performance degradation on unseen patients.
3. **Data Leakage in Literature**: Random window splitting across the same recording inflates cross-validation metrics.
4. **Black-box Predictions**: High-performing deep neural networks lack interpretability required for clinical decision support.

**NeuroGuard** resolves these challenges by providing:
- **Strict Patient-Independent Splitting**: Zero patient overlap between training, validation, and held-out test partitions.
- **Biomedical Signal Preprocessing**: Multi-channel bandpass (0.5–40 Hz) and 60 Hz notch filtering with fixed-duration sliding windows.
- **Comprehensive Benchmarking**: Classical machine learning (Logistic Regression, Random Forest, XGBoost) compared systematically against deep learning (1D CNN, CNN-LSTM).
- **Explainable AI (XAI)**: SHAP-based feature importance and gradient attribution to highlight frequency band contributions and seizure-active channels.
- **Full-Stack Deployment**: FastAPI REST backend, interactive Streamlit clinical dashboard, and containerization with Docker.

---

## 📂 2. Repository & Project Structure

The project is permanently maintained locally at:
`C:\Users\shree\OneDrive\Desktop\AML Cat 2 project\NeuroGuard_EEG_Seizure_Detection`

```
NeuroGuard_EEG_Seizure_Detection/
│
├── configs/
│   └── config.yaml             # Central reproducible pipeline configuration
├── data/
│   ├── raw/                    # Raw CHB-MIT EDF recordings and .summary.txt (gitignored)
│   ├── processed/              # Processed feature matrices and windowed tensors
│   └── interim/                # Intermediate filtered signals and cache
├── notebooks/                  # Prototyping and exploratory analysis
├── src/
│   ├── data/                   # Dataset acquisition, EDF parsing, and verification
│   ├── preprocessing/          # MNE filtering, normalization, window segmentation
│   ├── features/               # Time/frequency domain biomarker extraction
│   ├── models/                 # Classical baselines, 1D-CNN, and CNN-LSTM architectures
│   ├── training/               # Training pipelines, Optuna search, checkpointing
│   ├── evaluation/             # Patient-independent metrics, event analysis, PR-AUC
│   ├── explainability/         # SHAP, feature importances, Grad-CAM attribution
│   ├── visualization/          # Waveforms, power spectra, confusion matrices, UMAP
│   ├── api/                    # FastAPI application, Pydantic schemas, endpoints
│   └── utils/                  # Config loader, reproducibility seeds, logging
├── frontend/                   # Interactive Streamlit biomedical dashboard
├── tests/                      # Automated test suite (Pytest)
├── scripts/                    # Diagnostics, validation, and dataset download scripts
├── artifacts/
│   ├── models/                 # Saved model weights (.pt, .joblib)
│   ├── metrics/                # Evaluation reports and JSON metric logs
│   ├── plots/                  # Generated analytical figures and diagnostic plots
│   ├── reports/                # Benchmark summaries
│   └── mlruns/                 # MLflow experiment tracking logs
├── Dockerfile                  # Container definition for API and UI
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore               # Container build exclusions
├── .gitignore                  # Git exclusions (preventing data and weight leaks)
├── .env.example                # Template environment variables
├── pyproject.toml              # Build metadata and package dependencies
├── requirements.txt            # Dependency manifest
└── README.md                   # Project documentation and phase tracking
```

---

## 📊 3. Implementation Status & Phase Roadmap

| Phase | Milestone Description | Status | Verification / Artifacts |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Inspect environment & create permanent local project structure | **COMPLETED** | Verified local directories & hardware specs |
| **Phase 2** | Initialize configurations, requirements, diagnostics & baseline tests | **COMPLETED** | `config.yaml`, `validate_env.py`, `pytest` pass |
| **Phase 3** | Verify dataset access & inspect small subset of CHB-MIT EDFs | **COMPLETED** | Downloaded `chb01_03`, `chb02_16`, `chb03_01` & verified annotations |
| **Phase 4** | Implement preprocessing, annotation parsing & feature extraction | **COMPLETED** | Zero-phase filters, normalizer, 306 biomarkers, tensors in `data/processed/` |
| **Phase 5** | Train baseline classical models (Logistic Regression, Random Forest, XGBoost) | **COMPLETED** | LR, RF, XGBoost trained & saved; XGBoost Test ROC-AUC=0.9305, Recall=100% |
| **Phase 6** | Implement 1D CNN & comparative evaluation | **COMPLETED** | PyTorch 1D CNN checkpointed (`cnn1d_best.pt`); Test ROC-AUC=0.8111, Recall=85.19% |
| **Phase 7** | Implement CNN-LSTM spatio-temporal architecture | **COMPLETED** | CNN-LSTM checkpointed (`cnn_lstm_best.pt`); Test ROC-AUC=0.8379, PR-AUC=0.7751, Spec=85.33% |
| **Phase 8** | Patient-independent Optuna tuning, MLflow, XAI & 11 visualizations | **COMPLETED** | Bayesian Optuna tuning (Test ROC-AUC=0.9659, Rec=100%), 11 publication figures in `artifacts/plots/`, 29 unit tests pass |
| **Phase 9** | Implement FastAPI backend service | *In Progress / Next* | `/health`, `/predict`, `/explain`, `/model-info` endpoints |
| **Phase 10**| Implement Streamlit interactive research dashboard | *Pending* | Multi-page clinical waveform & inference UI |
| **Phase 11**| Containerize with Docker and Docker Compose | *Pending* | Local Docker build and container verification |
| **Phase 12**| Cloud deployment preparation (Hugging Face / AWS) | *Pending* | Deployment manifest and live service checklist |
| **Phase 13**| IEEE Academic Report & Final Submission Verification | *Pending* | 10-page academic report with empirical figures |

---

## 💻 4. Installation & Local Execution

### 4.1 Prerequisites
- Python 3.10+ (Tested on Python 3.13.7)
- Git 2.40+
- Minimum 10 GB free disk space (198.9 GB available on host)

### 4.2 Setup Environment
In Windows PowerShell:
```powershell
# Navigate to the project root
cd "C:\Users\shree\OneDrive\Desktop\AML Cat 2 project\NeuroGuard_EEG_Seizure_Detection"

# (Optional) Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 4.3 Run Environment Diagnostics
```powershell
python scripts/validate_env.py
```

### 4.4 Run Automated Tests
```powershell
pytest -v tests/
```

---

## 🔒 5. Research Rigor & Ethical Safeguards
- **Zero Patient Overlap**: Models are evaluated on patients never seen during training or validation.
- **No Data Fabrication**: Ground truth seizure onsets and offsets are parsed directly from official PhysioNet CHB-MIT annotation files.
- **Class-Imbalance Awareness**: Evaluation prioritizes Sensitivity/Recall, Specificity, F1-Score, and PR-AUC over naive accuracy.
- **Responsible AI**: Explainability heatmaps illustrate channel and spectral attributions as research aids, clearly separated from definitive diagnostic claims.
