# Cats vs Dogs MLOps Pipeline
[![CI Pipeline](https://github.com/your-username/cats-dogs-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/cats-dogs-mlops/actions/workflows/ci.yml)[![CD Pipeline](https://github.com/your-username/cats-dogs-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/your-username/cats-dogs-mlops/actions/workflows/cd.yml)[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://www.tensorflow.org/)[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
> A production-grade, end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) demonstrating the complete lifecycle: data ingestion, model training with experiment tracking, containerized API serving, CI/CD automation, GitOps deployment via Argo CD, and real-time monitoring with Grafana dashboards.
---
## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [End-to-End Flow](#end-to-end-flow)
  - [Phase 1 -- Data Pipeline](#phase-1----data-pipeline)
  - [Phase 2 -- Model Training & Experiment Tracking](#phase-2----model-training--experiment-tracking)
  - [Phase 3 -- API Packaging & Containerization](#phase-3----api-packaging--containerization)
  - [Phase 4 -- CI Pipeline](#phase-4----ci-pipeline)
  - [Phase 5 -- CD Pipeline (GitOps with Argo CD)](#phase-5----cd-pipeline-gitops-with-argo-cd)
  - [Phase 6 -- Monitoring & Observability](#phase-6----monitoring--observability)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Testing](#testing)
- [Makefile Reference](#makefile-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---
## Overview

This project implements a complete MLOps workflow that takes a machine learning model from prototype to production. It covers every stage of the ML lifecycle:

| Stage | Tools | Description |
|-------|-------|-------------|
| **Data Management** | DVC, Kaggle Hub | Version-controlled data pipeline with reproducible preprocessing |
| **Model Training** | TensorFlow/Keras, MLflow | CNN training with experiment tracking and model registry |
| **API Serving** | FastAPI, Uvicorn | High-performance REST API with async request handling |
| **Containerization** | Docker, multi-stage builds | Production-ready images with non-root user and health checks |
| **CI/CD** | GitHub Actions, GHCR | Automated linting, testing, building, and security scanning |
| **GitOps Deployment** | Argo CD, Kind, K8s | Declarative deployment with self-healing and auto-sync |
| **Monitoring** | Prometheus, Grafana | Real-time metrics, dashboards, and alerting rules |
| **Code Quality** | Pre-commit, Ruff, Black, MyPy | Automated code formatting and type checking |

---
## Architecture

### High-Level System Architecture

```
                          +---------------------------+
                          |     Developer Workstation  |
                          +-------------+-------------+
                                        |
                                   git push
                                        |
                          +-------------v-------------+
                          |     GitHub Repository      |
                          | (Source of Truth for GitOps)|
                          +------+------+------+------+
                                 |      |      |
                       +---------+  +---+---+  +----------+
                       |            |         |             |
                 +-----v----+  +---v---+  +--v-------+  +--v--------+
                 |  CI       |  |  CD   |  | Argo CD  |  |  DVC      |
                 | Pipeline  |  | Pipe  |  | (GitOps) |  |  Remote   |
                 +-----+----+  +---+---+  +----+-----+  +-----------+
                       |           |            |
           +-----------+-----------+            |
           |           |                        |
     +-----v-----+ +--v----------+      +------v-------+
     |  GHCR     | | Kind K8s    |      |  Kind K8s    |
     | (Docker   | | Cluster     |      |  Cluster     |
     | Registry) | +------+------+      +------+-------+
     +-----------+        |                    |
                    +-----v------+      +-----v-------+
                    | Cats vs Dogs|      | Monitoring  |
                    | API Service |      | Prometheus  |
                    | (Deployment)|      | + Grafana   |
                    +-----+------+      +------+-------+
                          |                    |
                   +------v------+      +------v-------+
                   |  Prometheus |      | Grafana Dash- |
                   |  /metrics   |<-----| board (14+   |
                   +-------------+      |  panels)     |
                                        +--------------+
```

### Data Flow Diagram

```
+------------------+     +------------------+     +------------------+
| Kaggle Dataset   |     | DVC Pipeline     |     | MLflow           |
| (Raw Images)     |---->| (Preprocessing)  |---->| (Experiments)    |
+------------------+     +------------------+     +------------------+
       |                        |                        |
       v                        v                        v
 data/raw/               data/processed/           mlflow.db
 (10,000 images)         data/train/               (SQLite)
                         data/val/
                         data/test/
                                |
                                v
                       +------------------+
                       | CNN Model        |
                       | (TensorFlow)     |
                       +------------------+
                                |
                                v
                       +------------------+
                       | FastAPI Service  |
                       | (REST API)       |
                       +------------------+
                                |
                                v
                       +------------------+
                       | Docker Container |
                       | (Production)     |
                       +------------------+
```

---

## End-to-End Flow

### Phase 1 -- Data Pipeline

The data pipeline downloads, preprocesses, and splits the Cats vs Dogs dataset using DVC for reproducibility and version control.

```
+-------------------+    +-------------------+    +-------------------+
| 1. DOWNLOAD       |    | 2. PREPROCESS     |    | 3. SPLIT          |
|                   |    |                   |    |                   |
| Kaggle Hub        |--->| Load images       |--->| Train: 80%        |
| Download dataset  |    | Resize to 224x224 |    | Val:   10%        |
| Copy to data/raw/ |    | Normalize [0,1]   |    | Test:  10%        |
|                   |    | Apply augment.    |    |                   |
| OUT: data/raw/    |    | Save as TFRecord  |    | OUT: data/train/  |
|                   |    | OUT: data/processed|   |      data/val/    |
+-------------------+    +-------------------+    |      data/test/   |
                                                  +-------------------+
```

**Key Details:**
- **Dataset**: `bhavikjikadara/dog-and-cat-classification-dataset` from Kaggle
- **Image Size**: 224x224 RGB (standard for CNN architectures)
- **Augmentation**: Random rotation (20 deg), zoom (0.2), horizontal flip, brightness adjustment
- **Format**: TFRecord for efficient I/O and sharding
- **DVC Tracking**: All data artifacts are version-controlled via DVC

```yaml
# dvc.yaml - Pipeline definition
stages:
  download:
    cmd: python src/data/download.py
    outs:
      - data/raw:
          cache: true
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw
      - src/data/preprocess.py
      - params.yaml
    outs:
      - data/processed:
          cache: true
  split:
    cmd: python src/data/split.py
    deps:
      - data/processed
      - src/data/split.py
      - params.yaml
    outs:
      - data/train:
          cache: true
      - data/val:
          cache: true
      - data/test:
          cache: true
```

### Phase 2 -- Model Training & Experiment Tracking

Training uses MLflow for experiment tracking, logging parameters, metrics, and model artifacts.

```
+-------------------+    +-------------------+    +-------------------+
| 4. CREATE MODEL   |    | 5. TRAIN          |    | 6. EVALUATE       |
|                   |    |                   |    |                   |
| CNN Architecture  |--->| Adam Optimizer    |--->| Test Set Metrics  |
| 4 Conv Blocks     |    | Early Stopping    |    | Confusion Matrix  |
| GlobalAvgPool     |    | ReduceLROnPlateau |    | Classification Rpt|
| Dense Classifier  |    | ModelCheckpoint   |    |                   |
|                   |    |                   |    | OUT: metrics,     |
| OUT: model arch.  |    | OUT: model.keras  |    |      plots        |
+-------------------+    +-------------------+    +-------------------+
         |                        |                        |
         v                        v                        v
+-------------------+    +-------------------+    +-------------------+
| 7. LOG TO MLFLOW  |    | 8. SAVE ARTIFACTS |    | 9. DVC PUSH       |
|                   |    |                   |    |                   |
| Parameters        |    | model.keras       |    | Version control   |
| Metrics           |    | metadata.json     |    | Push to remote    |
| Model signature   |    | training_history  |    |                   |
| Input examples    |    | confusion_matrix  |    | OUT: dvc remote   |
|                   |    | classification_rpt|    |                   |
| OUT: mlflow.db    |    |                   |    +-------------------+
+-------------------+    +-------------------+
```

**MLflow Tracking Configuration:**
```yaml
# params.yaml
mlflow:
  experiment_name: "cats-dogs-classification"
  tracking_uri: "sqlite:///mlflow.db"
  artifact_location: "mlflow_artifacts"
```

**Metrics Logged:**
- `test_accuracy`, `test_macro_f1`, `test_precision`, `test_recall`
- Per-class precision, recall, and F1 scores
- Training loss, accuracy, learning rate curves
- Confusion matrix and classification report

### Phase 3 -- API Packaging & Containerization

The trained model is served via a FastAPI REST API with structured logging and Prometheus metrics.

```
+-------------------+    +-------------------+    +-------------------+
| 10. FASTAPI APP   |    | 11. DOCKER BUILD  |    | 12. VALIDATE      |
|                   |    |                   |    |                   |
| /health           |    | Multi-stage build |    | Health check      |
| /predict          |<---| Non-root user     |<---| Endpoint test     |
| /predict/base64   |    | curl installed    |    | Prediction test   |
| /metrics          |    | Health check      |    | Metrics test      |
|                   |    |                   |    |                   |
| OUT: ASGI app     |    | OUT: Docker image |    | OUT: Pass/Fail    |
+-------------------+    +-------------------+    +-------------------+
```

**Docker Multi-Stage Build:**
```dockerfile
# Build stage - install dependencies
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage - minimal image
FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY src/ ./src/
COPY models/best_model/ ./models/best_model/
USER appuser
HEALTHCHECK CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Security Features:**
- Non-root user (`appuser`) in container
- No unnecessary packages installed
- Health check built into Dockerfile
- `.dockerignore` prevents sensitive files from being included

### Phase 4 -- CI Pipeline

The CI pipeline runs on every push and pull request with parallel jobs.

```
                   +-------------------+
                   |   git push / PR   |
                   +--------+----------+
                            |
                   +--------v----------+
                   |  CI Pipeline      |
                   |  (.github/         |
                   |   workflows/ci.yml)|
                   +--------+----------+
                            |
             +--------------+--------------+
             |              |              |
    +--------v------+ +----v-------+ +----v--------+
    | 1. LINT       | | 2. TEST    | | 3. DVC     |
    |               | |            | |   CHECK    |
    | Ruff (lint)   | | pytest     | |            |
    | Black (fmt)   | | coverage   | | dvc status |
    | MyPy (types)  | | 38 tests   | | validate   |
    +--------+------+ +----+-------+ +----+-------+
             |              |              |
             +--------------+--------------+
                            |
                   +--------v----------+
                   |  4. BUILD         |
                   |  Docker image     |
                   |  Push to GHCR     |
                   +--------+----------+
                            |
                   +--------v----------+
                   |  5. SECURITY      |
                   |  Trivy scan       |
                   |  SARIF upload     |
                   +-------------------+
```

**CI Jobs Breakdown:**

| Job | Runs On | Steps | Purpose |
|-----|---------|-------|---------|
| **lint** | ubuntu-latest | Ruff, Black, MyPy | Code quality and type safety |
| **test** | ubuntu-latest | pytest with coverage | Unit test validation (38 tests) |
| **dvc-check** | ubuntu-latest | dvc status | Pipeline structure validation |
| **build** | ubuntu-latest | Docker build + push | Container image creation |
| **security** | ubuntu-latest | Trivy filesystem scan | Vulnerability detection |

### Phase 5 -- CD Pipeline (GitOps with Argo CD)

The CD pipeline is triggered after CI succeeds and implements a full GitOps workflow.

```
+-------------------+    +-------------------+    +-------------------+
| 13. CI SUCCEEDS   |    | 14. BUILD & PUSH  |    | 15. GITOPS COMMIT |
|                   |    |                   |    |                   |
| workflow_run      |--->| Docker build      |--->| Update image tag  |
| completed         |    | Push to GHCR      |    | in k8s/deployment |
| on main           |    | Tag: git SHA      |    | git commit + push |
|                   |    |                   |    |                   |
|                   |    | OUT: GHCR image   |    | OUT: Git commit   |
+-------------------+    +-------------------+    +-------------------+
                                                         |
                                                         v
+-------------------+    +-------------------+    +-------------------+
| 18. SMOKE TESTS   |    | 17. ARGO CD SYNC  |    | 16. ARGO CD       |
|                   |    |                   |    |    DETECTS        |
| /health check  <--|----| Auto-sync to      |<---| Change in main    |
| /predict test     |    | Kind cluster      |    | branch            |
| /metrics test     |    | Self-healing on   |    | Auto-sync enabled |
|                   |    | Prune removed res |    |                   |
| OUT: Pass/Fail    |    | OUT: K8s deploy   |    |                   |
+-------------------+    +-------------------+    +-------------------+
```

**GitOps Principles Implemented:**

1. **Declarative**: All K8s manifests stored in Git (`k8s/` directory)
2. **Versioned**: Every deployment is a Git commit with unique image tag
3. **Automated**: Argo CD auto-syncs on Git changes
4. **Self-healing**: Manual cluster changes are reverted to Git state
5. **Pruning**: Resources removed from Git are deleted from cluster

**Argo CD Configuration:**
```yaml
# argocd/application.yaml
spec:
  syncPolicy:
    automated:
      prune: true        # Delete resources removed from Git
      selfHeal: true     # Revert manual changes
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 3           # Retry failed syncs
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 1m
```

### Phase 6 -- Monitoring & Observability

Real-time monitoring with Prometheus metrics, Grafana dashboards, and alerting rules.

```
+-------------------+    +-------------------+    +-------------------+
| 19. FASTAPI       |    | 20. PROMETHEUS    |    | 21. GRAFANA       |
|    /metrics       |    |                   |    |                   |
| 7 metric types    |--->| Scrape every 15s  |--->| 14+ dashboard     |
| Counter, Histogram|    | Store time series |    | panels            |
| Gauge             |    | Alert rules       |    | Auto-provisioned  |
|                   |    |                   |    |                   |
| OUT: metrics      |    | OUT: time series  |    | OUT: dashboards   |
+-------------------+    +-------------------+    +-------------------+
                                  |
                          +-------v--------+
                          | 22. ALERTS     |
                          |                |
                          | HighErrorRate  |
                          | HighLatency    |
                          | ModelNotLoaded |
                          | APIDown        |
                          | LowConfidence  |
                          +----------------+
```

**Prometheus Metrics Exposed:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `http_requests_in_progress` | Gauge | method, endpoint | In-flight requests |
| `model_predictions_total` | Counter | predicted_class | Predictions by class |
| `model_prediction_latency_seconds` | Histogram | -- | Inference latency |
| `model_prediction_confidence` | Histogram | predicted_class | Confidence distribution |
| `model_loaded` | Gauge | -- | Model availability (0/1) |

**Prometheus Alerting Rules:**

| Alert | Expression | Duration | Severity |
|-------|-----------|----------|----------|
| `HighErrorRate` | 5xx rate > 5% | 2 min | critical |
| `HighLatency` | P95 > 1s | 2 min | warning |
| `ModelNotLoaded` | `model_loaded == 0` | 1 min | critical |
| `APIDown` | No traffic | 2 min | critical |
| `LowModelConfidence` | Avg < 50% | 5 min | warning |

---

## Prerequisites

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| **Python** | 3.11+ | Runtime | `brew install python@3.11` |
| **Docker** | 24+ | Containerization | `brew install --cask docker` |
| **Kind** | 0.20+ | Local Kubernetes | `brew install kind` |
| **kubectl** | 1.28+ | K8s CLI | `brew install kubectl` |
| **DVC** | 3.40+ | Data versioning | `pip install dvc` |
| **Git** | 2.40+ | Version control | `brew install git` |
| **Argo CD CLI** | latest | GitOps | `brew install argocd` |

---

## Quick Start

### Option 1: Local Development (Fastest)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd cats-dogs-mlops

# 2. Create virtualenv and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
make install-dev

# 3. Initialize DVC
make dvc-init

# 4. Download and process data (requires Kaggle credentials)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
dvc repro download preprocess split

# 5. Train the model
dvc repro train

# 6. Start the API
make run

# 7. Test it
curl http://localhost:8000/health
```

### Option 2: Docker Compose (No Python needed)

```bash
# Build and run API
docker-compose up -d

# Or with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Access:
# API:        http://localhost:8000
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Option 3: Kubernetes with Argo CD (Full Production)

```bash
# 1. Create Kind cluster
kind create cluster --name mlops-cluster

# 2. Install Argo CD
make argocd-install

# 3. Deploy application
make argocd-deploy

# 4. Access services
kubectl port-forward -n argocd svc/argocd-server 8080:443
kubectl port-forward -n mlops-cats-dogs svc/cats-dogs-api 8000:8000

# 5. Deploy monitoring
make monitoring-k8s
kubectl port-forward -n monitoring svc/grafana 3000:3000
```

---

## Detailed Setup

### Kaggle Credentials

The data pipeline requires Kaggle API access:

```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Kaggle config file
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### DVC Remote Storage

DVC tracks data and model artifacts. Configure a remote for team sharing:

```bash
# Default: local remote at /tmp/dvc-store
make dvc-init

# For cloud storage (e.g., S3):
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# Push data to remote
dvc push
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `KAGGLE_USERNAME` | -- | Kaggle username for dataset download |
| `KAGGLE_KEY` | -- | Kaggle API key |

---

## API Reference

### Endpoints

#### `GET /` -- Root Info
```bash
curl http://localhost:8000/
```
```json
{
  "name": "Cats vs Dogs Classification API",
  "version": "1.0.0",
  "description": "Binary image classification API for Cats vs Dogs",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "predict_base64": "/predict/base64",
    "metrics": "/metrics"
  }
}
```

#### `GET /health` -- Health Check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_loaded": true
}
```

#### `POST /predict` -- Predict (File Upload)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cat_photo.jpg"
```
```json
{
  "class_name": "cat",
  "probability": 0.9234,
  "class_probabilities": {
    "cat": 0.9234,
    "dog": 0.0766
  },
  "model_version": "1.0.0"
}
```

#### `POST /predict/base64` -- Predict (Base64)
```bash
curl -X POST http://localhost:8000/predict/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."}'
```

#### `GET /metrics` -- Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```
Returns Prometheus text format metrics (7 metric types).


---

## Project Structure

```
cats-dogs-mlops/
|
|-- .github/workflows/           # GitHub Actions CI/CD
|   |-- ci.yml                   # CI: lint, test, build, security
|   |-- cd.yml                   # CD: build, push, deploy via Argo CD
|
|-- argocd/                      # Argo CD GitOps manifests
|   |-- project.yaml             # AppProject definition
|   |-- application.yaml         # Production Application
|   |-- application-staging.yaml # Staging Application
|   |-- namespace.yaml           # Argo CD namespace
|   |-- base/                    # Kustomize base
|   |   |-- kustomization.yaml
|   |-- overlays/                # Kustomize overlays
|       |-- production/
|       |   |-- kustomization.yaml
|       |-- staging/
|           |-- kustomization.yaml
|
|-- data/                        # DVC-tracked datasets
|   |-- raw/                     # Raw downloaded images (DVC)
|   |-- processed/               # Preprocessed TFRecords (DVC)
|   |-- train/                   # Training split (DVC)
|   |   |-- dataset.tfrecord
|   |   |-- metadata.json
|   |-- val/                     # Validation split (DVC)
|   |-- test/                    # Test split (DVC)
|
|-- docker/
|   |-- Dockerfile               # Multi-stage production build
|
|-- k8s/                         # Kubernetes manifests
|   |-- namespace.yaml           # mlops-cats-dogs namespace
|   |-- deployment.yaml          # API Deployment
|   |-- service.yaml             # ClusterIP Service
|   |-- configmap.yaml           # Configuration
|   |-- monitoring/              # Monitoring stack
|       |-- namespace.yaml       # monitoring namespace
|       |-- prometheus.yaml      # Prometheus + RBAC
|       |-- prometheus-configmap.yaml
|       |-- grafana.yaml         # Grafana Deployment + Service
|       |-- grafana-configmaps.yaml
|
|-- models/                      # DVC-tracked models
|   |-- best_model/
|       |-- model.keras          # Trained Keras model (DVC)
|       |-- metadata.json        # Model metadata for inference
|
|-- monitoring/                  # Monitoring configuration
|   |-- prometheus.yml           # Prometheus scrape config
|   |-- prometheus/
|   |   |-- alert_rules.yml      # Alerting rules (5 alerts)
|   |-- grafana/
|       |-- dashboards/
|       |   |-- mlops-api.json           # Auto-provisioned dashboard
|       |   |-- generate_dashboard.py    # Dashboard generator
|       |-- provisioning/
|           |-- dashboards/
|           |   |-- dashboards.yml
|           |-- datasources/
|               |-- prometheus.yml
|
|-- scripts/
|   |-- generate_synthetic_data.py  # Synthetic data for testing
|   |-- setup-argocd.sh            # Argo CD installation
|   |-- smoke_test.sh              # Post-deploy validation
|
|-- src/                         # Application source code
|   |-- __init__.py              # __version__ = "1.0.0"
|   |-- data/                    # Data pipeline modules
|   |   |-- download.py          # Kaggle dataset download
|   |   |-- preprocess.py        # Image preprocessing + augmentation
|   |   |-- split.py             # Train/val/test splitting
|   |-- models/                  # ML model modules
|   |   |-- cnn_model.py         # CNN architecture + compile
|   |   |-- train.py             # Training with MLflow tracking
|   |-- api/                     # FastAPI application
|   |   |-- main.py              # App, routes, middleware
|   |   |-- model_loader.py      # Model loading + inference
|   |   |-- schemas.py           # Pydantic request/response
|   |-- monitoring/
|   |   |-- metrics.py           # Prometheus metrics (7 types)
|   |-- utils/
|       |-- config.py            # YAML config management
|       |-- logging.py           # Structured JSON logging
|
|-- tests/                       # Unit tests (38 tests)
|   |-- conftest.py              # Shared fixtures
|   |-- test_config.py           # Config module tests (10)
|   |-- test_preprocess.py       # Preprocessing tests (7)
|   |-- test_model.py            # CNN model tests (7)
|   |-- test_inference.py        # Inference tests (8)
|   |-- test_monitoring.py       # Monitoring tests (6)
|
|-- .pre-commit-config.yaml      # Pre-commit hooks
|-- .dockerignore                # Docker build exclusions
|-- .gitignore                   # Git exclusions
|-- .dvcignore                   # DVC exclusions
|-- .dvc/                        # DVC configuration
|-- docker-compose.yml           # Docker Compose services
|-- dvc.yaml                     # DVC pipeline definition
|-- Makefile                     # Build/deploy targets
|-- model_architecture.json      # Model architecture JSON
|-- params.yaml                  # Hyperparameters
|-- reproduce.sh                 # DVC repro wrapper
|-- requirements.txt             # Production dependencies
|-- requirements-dev.txt         # Development dependencies
|-- training.log                 # Training CSV log
|-- README.md
```

---

## Configuration

All parameters are centralized in `params.yaml`:

```yaml
data:
  image_size: 224            # Input image dimensions (224x224)
  batch_size: 32             # Training batch size
  train_split: 0.8           # 80% training data
  val_split: 0.1             # 10% validation data
  test_split: 0.1            # 10% test data
  augmentation:
    rotation_range: 20       # Random rotation degrees
    zoom_range: 0.2          # Random zoom range
    horizontal_flip: true    # Random horizontal flip
    brightness_range: [0.8, 1.2]  # Brightness adjustment

model:
  type: "cnn"                # Model architecture type
  learning_rate: 0.001       # Adam optimizer learning rate
  epochs: 20                 # Maximum training epochs
  early_stopping_patience: 5 # Stop if no improvement for 5 epochs
  reduce_lr_patience: 3      # Reduce LR if no improvement for 3 epochs
  reduce_lr_factor: 0.5      # Multiply LR by 0.5 on plateau

mlflow:
  experiment_name: "cats-dogs-classification"
  tracking_uri: "sqlite:///mlflow.db"
  artifact_location: "mlflow_artifacts"
```

### Configuration Access in Code

```python
from src.utils.config import get_config

# Get full config object
config = get_config()
print(config.data.image_size)  # 224

# Get specific value with dot notation
batch_size = get_config("data.batch_size")  # 32

# Get with default
lr = get_config("model.nonexistent", default=0.01)  # 0.01
```

---

## Model Architecture

### CNN Architecture (4-Block Design)

```
Input (224, 224, 3)
    |
    v
+------------------------------------------+
| Block 1: 32 filters                       |
| Conv2D(32, 3x3) -> BN -> Conv2D(32, 3x3) -> BN |
| MaxPool(2x2) -> Dropout(0.25)            |
+------------------------------------------+   Output: (112, 112, 32)
    |
    v
+------------------------------------------+
| Block 2: 64 filters                       |
| Conv2D(64, 3x3) -> BN -> Conv2D(64, 3x3) -> BN |
| MaxPool(2x2) -> Dropout(0.25)            |
+------------------------------------------+   Output: (56, 56, 64)
    |
    v
+------------------------------------------+
| Block 3: 128 filters                      |
| Conv2D(128, 3x3) -> BN -> Conv2D(128, 3x3) -> BN |
| MaxPool(2x2) -> Dropout(0.25)            |
+------------------------------------------+   Output: (28, 28, 128)
    |
    v
+------------------------------------------+
| Block 4: 256 filters                      |
| Conv2D(256, 3x3) -> BN                    |
| MaxPool(2x2) -> Dropout(0.25)            |
+------------------------------------------+   Output: (14, 14, 256)
    |
    v
+------------------------------------------+
| Classifier                                |
| GlobalAveragePooling2D -> (256)           |
| Dense(512, relu) -> BN -> Dropout(0.5)    |
| Dense(256, relu) -> BN -> Dropout(0.5)    |
| Dense(2, softmax) -> (2,)                 |
+------------------------------------------+
```

**Training Configuration:**
- **Total Parameters**: ~3.5M
- **Loss**: `SparseCategoricalCrossentropy`
- **Optimizer**: Adam (lr=0.001)
- **Metrics**: `SparseCategoricalAccuracy`, `SparseTopKCategoricalAccuracy(k=2)`

**Training Callbacks:**
- `EarlyStopping`: Stops training if `val_loss` does not improve for 5 epochs, restores best weights
- `ModelCheckpoint`: Saves best model based on `val_accuracy` (max)
- `ReduceLROnPlateau`: Reduces learning rate by factor 0.5 if `val_loss` does not improve for 3 epochs
- `CSVLogger`: Logs training metrics to `training.log`

### Training Flow

```
train_ds (80%)              val_ds (10%)               test_ds (10%)
    |                           |                           |
    v                           v                           |
+---+---------------------------+---+                       |
|          model.fit()              |                       |
|                                   |                       |
|  Epoch 1: loss=0.69, acc=0.52    |                       |
|  Epoch 2: loss=0.58, acc=0.68    |                       |
|  ...                              |                       |
|  Epoch N: (early stopping)       |                       |
+---+-------------------------------+                       |
    |                                                       |
    v                                                       |
Best model checkpoint (model.keras)                         |
    |                                                       |
    v                                                       |
evaluate_model() <------------------------------------------+
    |
    v
+---+---+---+---+
| accuracy    |  (e.g., 0.95)
| macro_f1    |  (e.g., 0.94)
| precision   |  (per class)
| recall      |  (per class)
| confusion   |  (2x2 matrix)
+-------------+
```

---

## Testing

### Test Suite Overview

The project includes 38 unit tests across 6 test files:

```
tests/
|-- conftest.py            # Shared fixtures (sample images, paths)
|-- test_config.py         # 10 tests - Configuration management
|-- test_preprocess.py     # 7 tests  - Data preprocessing pipeline
|-- test_model.py          # 7 tests  - CNN model architecture
|-- test_inference.py      # 8 tests  - Model inference & API schemas
|-- test_monitoring.py     # 6 tests  - Prometheus metrics
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_config.py -v

# Run with verbose output and coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test
pytest tests/test_model.py::TestCNNModel::test_create_cnn_model_default -v
```

### Test Coverage by Module

| Module | Test File | Tests | Coverage Area |
|--------|-----------|-------|---------------|
| `src/utils/config.py` | `test_config.py` | 10 | Config loading, dot notation, caching |
| `src/data/preprocess.py` | `test_preprocess.py` | 7 | Image loading, augmentation, TFRecord |
| `src/models/cnn_model.py` | `test_model.py` | 7 | Model creation, compilation, callbacks |
| `src/api/model_loader.py` | `test_inference.py` | 8 | Preprocessing, base64, prediction |
| `src/monitoring/metrics.py` | `test_monitoring.py` | 6 | HTTP/prediction metrics, model status |

### Test Fixtures

```python
# conftest.py - Shared fixtures
@pytest.fixture(scope="session")
def sample_image_bytes():
    """Create a 224x224 red JPEG as bytes."""
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture(scope="session")
def sample_image_file():
    """Create a 224x224 red JPEG as a temp file."""
    img = Image.new("RGB", (224, 224), color="red")
    path = Path(tempfile.mktemp(suffix=".jpg"))
    img.save(path)
    yield path
    path.unlink()
```

---

## Makefile Reference

```bash
make help              # Show all available targets
```

### Development Targets

| Target | Description | Command |
|--------|-------------|---------|
| `install-dev` | Install all dependencies (prod + dev) | `pip install -r requirements.txt -r requirements-dev.txt` |
| `test` | Run unit tests with coverage | `pytest tests/ -v --cov=src --cov-report=term-missing` |
| `lint` | Run Ruff, Black, MyPy | `ruff check` + `black --check` + `mypy` |
| `run` | Start API locally with hot-reload | `uvicorn src.api.main:app --reload --port 8000` |
| `mlflow-ui` | Start MLflow tracking UI | `mlflow ui --port 5000` |

### DVC Targets

| Target | Description | Command |
|--------|-------------|---------|
| `dvc-init` | Initialize DVC with local remote | `dvc init && dvc remote add` |
| `dvc-pull` | Pull data/models from DVC remote | `dvc pull` |
| `dvc-push` | Push data/models to DVC remote | `dvc push` |
| `dvc-repro` | Reproduce full DVC pipeline | `dvc repro` |
| `train` | Train model (alias for dvc-repro) | `dvc repro train` |

### Deployment Targets

| Target | Description | Command |
|--------|-------------|---------|
| `build` | Build Docker image | `docker build -f docker/Dockerfile -t cats-dogs-api:latest .` |
| `deploy-kind` | Deploy to Kind cluster via kubectl | Creates namespace, applies K8s manifests |
| `argocd-install` | Install Argo CD on cluster | Runs `scripts/setup-argocd.sh` |
| `argocd-deploy` | Deploy app via Argo CD | Applies project + application YAML |
| `argocd-status` | Show Argo CD app status | `argocd app get` |
| `smoke-test` | Run post-deploy smoke tests | Runs `scripts/smoke_test.sh` |

### Monitoring Targets

| Target | Description | Command |
|--------|-------------|---------|
| `monitoring-up` | Start Prometheus + Grafana | `docker-compose --profile monitoring up -d` |
| `monitoring-down` | Stop monitoring stack | `docker-compose --profile monitoring down` |
| `monitoring-k8s` | Deploy monitoring to Kind | Applies K8s monitoring manifests |

### Utility Targets

| Target | Description | Command |
|--------|-------------|---------|
| `clean` | Clean generated files | Removes `__pycache__`, `.pytest_cache`, `mlflow.db`, etc. |

---

## Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow Installation Fails

```bash
# Problem: TensorFlow 2.15 requires Python 3.11
python3 --version  # Check your Python version

# Solution: Use Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Kaggle Dataset Download Fails

```bash
# Problem: Missing Kaggle credentials
# Solution: Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Or verify credentials
kaggle datasets list -s "dog-and-cat"
```

#### 3. DVC Pipeline Error: `pathspec` Import

```bash
# Problem: DVC 3.40.0 has a known pathspec issue
# Solution: Upgrade DVC
pip install "dvc>=3.40.0"
# DVC 3.67.1+ resolves this
```

#### 4. Docker Build Fails: Model Not Found

```bash
# Problem: models/best_model/model.keras not in build context
# Solution: Train first, then build
dvc repro train
docker build -f docker/Dockerfile -t cats-dogs-api:latest .
```

#### 5. Port Already in Use

```bash
# Problem: Port 8000 is occupied
# Solution: Kill existing process or use different port
lsof -ti:8000 | xargs kill -9
# Or run on different port
uvicorn src.api.main:app --port 8001
```

#### 6. Argo CD Application stuck in "Progressing"

```bash
# Problem: Image pull or container crash
# Solution: Check pod status
kubectl get pods -n mlops-cats-dogs
kubectl describe pod <pod-name> -n mlops-cats-dogs
kubectl logs <pod-name> -n mlops-cats-dogs

# Check Argo CD sync status
argocd app get cats-dogs-api-production
argocd app sync cats-dogs-api-production --force
```

#### 7. Grafana Dashboard Not Loading

```bash
# Problem: Dashboard provisioning not working
# Solution: Verify volume mounts
docker exec -it grafana ls /var/lib/grafana/dashboards/
# Should show mlops-api.json

# Check Grafana logs
docker logs grafana | grep -i dashboard
```

#### 8. Pre-commit Hooks Failing

```bash
# Problem: Formatting issues
# Solution: Run formatters manually
ruff check src/ tests/ --fix
black src/ tests/
# Then commit again
```

### Getting Help

```bash
# Check all available commands
make help

# View DVC pipeline status
dvc status

# Check Docker Compose services
docker-compose ps

# View K8s resources
kubectl get all -n mlops-cats-dogs
kubectl get all -n monitoring
```

---

## Complete GitOps Deployment Flow

This diagram shows the entire lifecycle from code commit to production deployment:

```
Developer commits code
        |
        v
+-------+--------+
| GitHub Push     |
+-------+--------+
        |
        +---> CI Pipeline (ci.yml)
        |     |
        |     +---> Lint (Ruff, Black, MyPy)
        |     +---> Test (pytest, 38 tests)
        |     +---> DVC Check (pipeline validation)
        |     +---> Build Docker Image
        |     +---> Push to GHCR (ghcr.io/owner/cats-dogs-mlops:sha-xxxx)
        |     +---> Security Scan (Trivy)
        |     |
        |     v
        |  CI Success
        |
        +---> CD Pipeline (cd.yml) triggered
              |
              +---> Build & Push Docker Image (with git SHA tag)
              +---> GitOps Commit
              |     |
              |     +---> sed -i update k8s/deployment.yaml image tag
              |     +---> git commit + push to main
              |     |
              |     v
              |  Argo CD detects Git change
              |
              +---> Argo CD Sync
              |     |
              |     +---> Apply k8s/namespace.yaml
              |     +---> Apply k8s/deployment.yaml
              |     +---> Apply k8s/service.yaml
              |     +---> Apply k8s/configmap.yaml
              |     |
              |     v
              |  Kind K8s Cluster updated
              |
              +---> Smoke Tests
                    |
                    +---> GET /health (200 OK)
                    +---> POST /predict/base64 (class_name returned)
                    +---> GET /metrics (http_requests_total)
                    |
                    v
               Deployment Complete!
```

### Key Ports Reference

| Service | Port | Access |
|---------|------|--------|
| **API** | 8000 | `http://localhost:8000` |
| **MLflow UI** | 5000 | `http://localhost:5000` |
| **Prometheus** | 9090 | `http://localhost:9090` |
| **Grafana** | 3000 | `http://localhost:3000` (admin/admin) |
| **Argo CD** | 8080 | `https://localhost:8080` |

### K8s Namespace Layout

```
Kind Cluster: mlops-cluster
|
|-- argocd              # Argo CD control plane
|   |-- argocd-server
|   |-- argocd-repo-server
|   |-- argocd-application-controller
|
|-- mlops-cats-dogs     # Application namespace
|   |-- cats-dogs-api-deployment
|   |-- cats-dogs-api-service (ClusterIP:8000)
|
|-- monitoring          # Monitoring namespace
    |-- prometheus-deployment (NodePort:30090)
    |-- grafana-deployment (NodePort:30030)
```

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
