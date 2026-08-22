#!/bin/bash
# setup-argocd.sh - Install and configure Argo CD on a Kubernetes cluster
# Usage: ./scripts/setup-argocd.sh [namespace] [version]

set -euo pipefail

ARGOCD_NAMESPACE="${1:-argocd}"
ARGOCD_VERSION="${2:-stable}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
for cmd in kubectl; do
    if ! command -v $cmd &> /dev/null; then
        log_error "$cmd is not installed"
        exit 1
    fi
done

log_info "Installing Argo CD (version: $ARGOCD_VERSION)..."

# Create namespace
kubectl create namespace $ARGOCD_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Install Argo CD
kubectl apply -n $ARGOCD_NAMESPACE -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/install.yaml

# Wait for Argo CD to be ready
log_info "Waiting for Argo CD server to be ready..."
kubectl wait --for=condition=available deployment/argocd-server \
    -n $ARGOCD_NAMESPACE \
    --timeout=300s

# Get initial admin password
ARGOCD_PASSWORD=$(kubectl -n $ARGOCD_NAMESPACE get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || echo "")

if [ -n "$ARGOCD_PASSWORD" ]; then
    log_info "Argo CD initial admin password: $ARGOCD_PASSWORD"
else
    log_warn "Could not retrieve initial admin password"
fi

# Install argocd CLI (if not present)
if ! command -v argocd &> /dev/null; then
    log_info "Installing Argo CD CLI..."
    curl -sSL -o /usr/local/bin/argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) 2>/dev/null || \
    brew install argocd 2>/dev/null || \
    log_warn "Could not install argocd CLI automatically. Install manually."
fi

# Create the AppProject
log_info "Creating Argo CD project..."
kubectl apply -f argocd/project.yaml 2>/dev/null || \
    log_warn "Could not apply project.yaml (may need Argo CD running first)"

# Create the Application
log_info "Creating Argo CD application..."
kubectl apply -f argocd/application.yaml 2>/dev/null || \
    log_warn "Could not apply application.yaml (may need project first)"

# Port forward Argo CD UI
log_info "Setting up port-forward for Argo CD UI..."
kubectl port-forward -n $ARGOCD_NAMESPACE svc/argocd-server 8080:443 &
PF_PID=$!

sleep 5

log_info "=========================================="
log_info "Argo CD Installation Complete!"
log_info "=========================================="
log_info "Argo CD UI:    https://localhost:8080"
log_info "Username:      admin"
log_info "Password:      $ARGOCD_PASSWORD"
log_info "Port Forward:  PID $PF_PID"
log_info ""
log_info "To login via CLI:"
log_info "  argocd login localhost:8080 --username admin --password $ARGOCD_PASSWORD --insecure"
log_info ""
log_info "To deploy the application:"
log_info "  kubectl apply -f argocd/application.yaml"
log_info "=========================================="
