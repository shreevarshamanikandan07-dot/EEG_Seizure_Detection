
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Install project dependencies only once.
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install CPU-only PyTorch separately.
RUN pip install --no-cache-dir \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Copy application files and existing model artifacts.
COPY configs/ ./configs/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY README.md .

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
