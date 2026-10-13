# NeuroGuard Deployment & Production Serving Guide

This document specifies the deployment architecture and execution procedures for the **NeuroGuard Explainable EEG Seizure Detection** system across local containerized environments, Hugging Face Spaces, and cloud virtual machines (AWS EC2).

---

## 1. Local Containerized Deployment (Docker & Docker Compose)

NeuroGuard provides a multi-container architecture isolating the FastAPI inference backend and the Streamlit clinical decision support dashboard.

### 1.1 Prerequisites
- Docker Engine 24.0+
- Docker Compose v2.20+
- Minimum 4 GB RAM allocated to Docker Daemon

### 1.2 Execution Commands
```bash
# Clone and enter directory
cd NeuroGuard_EEG_Seizure_Detection

# Build and start services in background
docker compose up --build -d

# Check running container status
docker compose ps

# Inspect logs
docker compose logs -f api
docker compose logs -f frontend
```

### 1.3 Service URLs
- **FastAPI REST API & Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Streamlit Interactive Clinical Dashboard**: [http://localhost:8501](http://localhost:8501)

### 1.4 Teardown
```bash
docker compose down --volumes
```

---

## 2. Hugging Face Spaces Deployment (Streamlit App)

Hugging Face Spaces allows zero-cost cloud hosting for biomedical research prototypes.

### 2.1 Space Creation
1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: `neuroguard-eeg-seizure-detection`.
3. Select SDK: **Streamlit**.
4. Choose Hardware: **CPU Basic (2 vCPU, 16 GB RAM)** — Free tier.

### 2.2 Hugging Face Repository Configuration
Include a top-level `README.md` with metadata YAML block:
```yaml
---
title: NeuroGuard EEG Seizure Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.64.0
app_file: frontend/app.py
pinned: false
---
```

### 2.3 Upload and Sync
```bash
# Add Hugging Face remote
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/neuroguard-eeg-seizure-detection

# Push project
git push space master:main
```

---

## 3. AWS EC2 Production Cloud Deployment

For enterprise clinical research testbeds requiring dedicated compute and custom domains.

### 3.1 Recommended Instance Sizing
- **Instance Type**: `t3.large` (2 vCPUs, 8 GiB RAM) or `c6i.large` (compute-optimized).
- **Storage**: 30 GiB gp3 EBS volume.
- **Operating System**: Ubuntu Server 22.04 LTS (HVM).

### 3.2 Security Group Firewall Configuration
| Protocol | Port Range | Source | Purpose |
| :--- | :--- | :--- | :--- |
| TCP | 22 | Administrator IP | SSH Remote Management |
| TCP | 80 | 0.0.0.0/0 | HTTP (Redirect to HTTPS) |
| TCP | 443 | 0.0.0.0/0 | HTTPS TLS/SSL Traffic |
| TCP | 8000 | VPC / Private | Direct API (Internal) |
| TCP | 8501 | VPC / Private | Direct UI (Internal) |

### 3.3 Server Provisioning Script
```bash
# Update and install system dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git nginx certbot python3-certbot-nginx

# Clone repository
git clone https://github.com/shree-workspace/NeuroGuard_EEG_Seizure_Detection.git
cd NeuroGuard_EEG_Seizure_Detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Systemd Daemon Configuration

Create `/etc/systemd/system/neuroguard-api.service`:
```ini
[Unit]
Description=NeuroGuard FastAPI Inference Daemon
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/NeuroGuard_EEG_Seizure_Detection
ExecStart=/home/ubuntu/NeuroGuard_EEG_Seizure_Detection/venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/neuroguard-ui.service`:
```ini
[Unit]
Description=NeuroGuard Streamlit Dashboard Daemon
After=network.target neuroguard-api.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/NeuroGuard_EEG_Seizure_Detection
ExecStart=/home/ubuntu/NeuroGuard_EEG_Seizure_Detection/venv/bin/streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start daemons:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now neuroguard-api
sudo systemctl enable --now neuroguard-ui
```

### 3.5 NGINX Reverse Proxy & SSL Configuration
Add to `/etc/nginx/sites-available/neuroguard`:
```nginx
server {
    server_name neuroguard.yourdomain.org;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable site and acquire SSL certificate:
```bash
sudo ln -s /etc/nginx/sites-available/neuroguard /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d neuroguard.yourdomain.org
```

---

## 4. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Environment tier (`development`, `staging`, `production`) |
| `API_HOST` | `127.0.0.1` | Binding address for FastAPI |
| `API_PORT` | `8000` | Port for FastAPI service |
| `BACKEND_API_URL` | `http://127.0.0.1:8000` | Backend API URL for Streamlit frontend |
| `MODEL_DEVICE` | `cpu` | PyTorch inference device (`cpu` or `cuda`) |
| `LOG_LEVEL` | `INFO` | Python logging severity level |

---

## 5. Security & Clinical Responsibility
- **Data Protection**: Patient EEG timeseries must be transmitted exclusively across encrypted TLS/SSL channels.
- **De-identification**: Ensure all input recordings strictly exclude PHI (Protected Health Information) matching HIPAA / GDPR compliance.
- **Regulatory Status**: Display mandatory research disclaimer on all public and internal interfaces.
