# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential, libgomp1 for XGBoost/OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir filelock typing-extensions sympy networkx fsspec
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu torch# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY README.md .

# Expose ports: 8000 for FastAPI backend, 8501 for Streamlit frontend
EXPOSE 8000 8501

# Default command: launch FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
