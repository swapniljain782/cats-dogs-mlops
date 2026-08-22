#!/bin/bash
# smoke_test.sh - Post-deployment smoke tests for M4
# Runs health check and prediction tests against a deployed service

set -euo pipefail

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

FAILED=0

# Test 1: Health Check
echo "=== Smoke Test 1: Health Check ==="
HEALTH_OK=false
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        HEALTH_RESPONSE=$(curl -s "${API_URL}/health")
        log_info "Health check passed (attempt $i)"
        echo "Response: $HEALTH_RESPONSE"
        HEALTH_OK=true
        break
    fi
    log_warn "Waiting for health endpoint... ($i/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done

if [ "$HEALTH_OK" != "true" ]; then
    log_error "Health check failed after $MAX_RETRIES attempts"
    FAILED=1
fi

# Test 2: Root endpoint
echo ""
echo "=== Smoke Test 2: Root Endpoint ==="
ROOT_RESPONSE=$(curl -sf "${API_URL}/" 2>/dev/null || echo "")
if echo "$ROOT_RESPONSE" | grep -q "Cats vs Dogs"; then
    log_info "Root endpoint passed"
else
    log_error "Root endpoint failed"
    FAILED=1
fi

# Test 3: Prediction endpoint
echo ""
echo "=== Smoke Test 3: Prediction Endpoint ==="
# Generate test image as base64
TEST_IMAGE_B64=$(python3 -c "
from PIL import Image
import io
import base64
img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
print(base64.b64encode(buf.getvalue()).decode())
" 2>/dev/null || echo "")

if [ -z "$TEST_IMAGE_B64" ]; then
    log_warn "Could not generate test image via Python, using skip"
else
    PREDICTION=$(curl -sf -X POST "${API_URL}/predict/base64" \
        -H "Content-Type: application/json" \
        -d "{\"image_base64\": \"${TEST_IMAGE_B64}\"}" 2>/dev/null || echo "")
    
    if echo "$PREDICTION" | grep -q "class_name"; then
        log_info "Prediction endpoint passed"
        echo "Response: $PREDICTION"
    else
        log_error "Prediction endpoint failed"
        FAILED=1
    fi
fi

# Test 4: Metrics endpoint
echo ""
echo "=== Smoke Test 4: Metrics Endpoint ==="
METRICS=$(curl -sf "${API_URL}/metrics" 2>/dev/null || echo "")
if echo "$METRICS" | grep -q "http_requests_total"; then
    log_info "Metrics endpoint passed"
else
    log_warn "Metrics endpoint returned unexpected format (non-critical)"
fi

# Summary
echo ""
echo "=========================="
if [ $FAILED -eq 0 ]; then
    log_info "All smoke tests PASSED"
    exit 0
else
    log_error "Some smoke tests FAILED"
    exit 1
fi
