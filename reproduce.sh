#!/bin/bash
# reproduce.sh - End-to-end reproduction script for Cats vs Dogs MLOps Pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    for cmd in python3 docker kind kubectl dvc git; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    log_info "All prerequisites found"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Python dependencies
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt -r requirements-dev.txt
    
    # DVC initialization
    if [ ! -d ".dvc" ]; then
        dvc init
        dvc remote add -d localremote /tmp/dvc-store
    fi
    
    log_info "Environment setup complete"
}

# Run data pipeline
run_data_pipeline() {
    log_info "Running data pipeline..."
    
    if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
        log_warn "Kaggle credentials not set. Using synthetic data for demo."
        python3 scripts/generate_synthetic_data.py
    else
        dvc repro download preprocess split
    fi
    
    log_info "Data pipeline complete"
}

# Train model
train_model() {
    log_info "Training model..."
    
    dvc repro train
    
    log_info "Model training complete"
}

# Start MLflow UI
start_mlflow() {
    log_info "Starting MLflow UI..."
    
    mlflow ui --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlflow_artifacts \
        --host 0.0.0.0 --port 5000 &
    
    MLFLOW_PID=$!
    sleep 5
    
    log_info "MLflow UI running at http://localhost:5000 (PID: $MLFLOW_PID)"
}

# Build Docker image
build_docker() {
    log_info "Building Docker image..."
    
    docker build -f docker/Dockerfile -t cats-dogs-api:latest .
    
    log_info "Docker image built"
}

# Run API locally
run_api_local() {
    log_info "Running API locally..."
    
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API running at http://localhost:8000 (PID: $API_PID)"
    else
        log_error "API failed to start"
        kill $API_PID
        exit 1
    fi
}

# Test API
test_api() {
    log_info "Testing API..."
    
    # Health check
    curl -s http://localhost:8000/health | python3 -m json.tool
    
    # Create test image
    python3 -c "
from PIL import Image
import io
import base64
img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
print(base64.b64encode(buf.getvalue()).decode())
" > /tmp/test_image.b64
    
    # Test prediction
    curl -s -X POST http://localhost:8000/predict/base64 \
        -H "Content-Type: application/json" \
        -d @"/tmp/test_image.b64" | python3 -m json.tool
    
    log_info "API tests passed"
}

# Deploy to Kind with Argo CD
deploy_kind_argocd() {
    log_info "Deploying to Kind with Argo CD..."
    
    # Create cluster if not exists
    if ! kind get clusters 2>/dev/null | grep -q "mlops-cluster"; then
        kind create cluster --name mlops-cluster --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
      - containerPort: 443
        hostPort: 443
EOF
    fi
    
    # Install ingress-nginx
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=120s
    
    # Install Argo CD
    bash scripts/setup-argocd.sh
    
    log_info "Argo CD deployment complete"
}

# Deploy to Kind with kubectl (direct)
deploy_kind_kubectl() {
    log_info "Deploying to Kind with kubectl..."
    
    # Create cluster if not exists
    if ! kind get clusters 2>/dev/null | grep -q "mlops-cluster"; then
        kind create cluster --name mlops-cluster --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
      - containerPort: 443
        hostPort: 443
EOF
    fi
    
    # Install ingress-nginx
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=120s
    
    # Deploy manifests directly
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    
    # Wait for deployment
    kubectl rollout status deployment/cats-dogs-api -n mlops-cats-dogs --timeout=300s
    
    log_info "kubectl deployment complete"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Port forward
    kubectl port-forward -n mlops-cats-dogs svc/cats-dogs-api 8000:8000 &
    PF_PID=$!
    sleep 10
    
    # Health check
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_info "Health check passed"
            break
        fi
        sleep 2
    done
    
    # Prediction test
    python3 -c "
from PIL import Image
import io
import base64
img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
print(base64.b64encode(buf.getvalue()).decode())
" > /tmp/test_image.b64
    
    PREDICTION=$(curl -s -X POST http://localhost:8000/predict/base64 \
        -H "Content-Type: application/json" \
        -d @"/tmp/test_image.b64")
    
    if echo "$PREDICTION" | grep -q "class_name"; then
        log_info "Prediction test passed"
    else
        log_error "Prediction test failed"
        kill $PF_PID
        exit 1
    fi
    
    kill $PF_PID
    log_info "All smoke tests passed"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Kill background processes
    pkill -f "mlflow ui" || true
    pkill -f "uvicorn" || true
    pkill -f "kubectl port-forward" || true
    
    log_info "Cleanup complete"
}

# Main execution
main() {
    log_info "Starting Cats vs Dogs MLOps Pipeline Reproduction"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    check_prerequisites
    setup_environment
    run_data_pipeline
    train_model
    
    # Start services for demo
    start_mlflow
    run_api_local
    test_api
    
    # Optional: Deploy to Kind
    DEPLOY_METHOD="${DEPLOY_METHOD:-false}"
    
    if [ "$DEPLOY_METHOD" = "argocd" ]; then
        build_docker
        deploy_kind_argocd
        run_smoke_tests
    elif [ "$DEPLOY_METHOD" = "kubectl" ]; then
        build_docker
        deploy_kind_kubectl
        run_smoke_tests
    elif [ "$DEPLOY_METHOD" = "true" ]; then
        # Legacy support
        build_docker
        deploy_kind_kubectl
        run_smoke_tests
    fi
    
    log_info "Reproduction complete!"
    log_info "MLflow UI: http://localhost:5000"
    log_info "API: http://localhost:8000"
    log_info "API Docs: http://localhost:8000/docs"
    if [ "$DEPLOY_METHOD" = "argocd" ]; then
        log_info "Argo CD UI: https://localhost:8080"
    fi
    
    # Keep running for demo
    wait
}

# Run main
main "$@"
