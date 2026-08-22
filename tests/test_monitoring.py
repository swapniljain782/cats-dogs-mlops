"""Unit tests for monitoring and metrics functions."""
import pytest
from src.monitoring.metrics import (
    record_http_request,
    record_prediction,
    set_model_loaded,
    get_metrics,
    http_requests_total,
    model_predictions_total,
    model_loaded,
)
from prometheus_client import generate_latest


class TestMetrics:
    """Tests for Prometheus metrics."""
    
    def test_record_http_request(self):
        """Test recording HTTP request metrics."""
        # Should not raise any exceptions
        record_http_request("GET", "/health", 200, 0.05)
        record_http_request("POST", "/predict", 200, 0.15)
        record_http_request("POST", "/predict", 500, 0.30)
        
        # Verify metrics are recorded
        metrics_output = get_metrics()
        assert isinstance(metrics_output, bytes)
        assert b"http_requests_total" in metrics_output
    
    def test_record_prediction(self):
        """Test recording prediction metrics."""
        record_prediction("cat", 0.05, 0.95)
        record_prediction("dog", 0.08, 0.87)
        
        metrics_output = get_metrics()
        assert b"model_predictions_total" in metrics_output
        assert b"model_prediction_latency_seconds" in metrics_output
    
    def test_set_model_loaded(self):
        """Test setting model loaded status."""
        set_model_loaded(True, "1.0.0")
        
        metrics_output = get_metrics()
        assert b"model_loaded" in metrics_output
        assert b"model_version_info" in metrics_output
    
    def test_set_model_not_loaded(self):
        """Test setting model not loaded status."""
        set_model_loaded(False)
        
        metrics_output = get_metrics()
        assert b"model_loaded" in metrics_output
    
    def test_get_metrics_format(self):
        """Test that get_metrics returns Prometheus-compatible format."""
        # Record some metrics first
        record_http_request("GET", "/health", 200, 0.01)
        record_prediction("cat", 0.02, 0.99)
        
        metrics = get_metrics()
        assert isinstance(metrics, bytes)
        
        # Should be decodable as text
        metrics_text = metrics.decode("utf-8")
        assert "http_requests_total" in metrics_text
        assert "model_prediction_latency_seconds" in metrics_text
    
    def test_metrics_multiple_requests(self):
        """Test metrics accumulate correctly."""
        for _ in range(5):
            record_http_request("GET", "/health", 200, 0.01)
        
        metrics = get_metrics().decode("utf-8")
        # Find the counter line and check it incremented
        for line in metrics.split("\n"):
            if line.startswith("http_requests_total") and 'method="GET"' in line and 'endpoint="/health"' in line and 'status="200"' in line:
                # Extract the value
                value = float(line.split()[-1])
                assert value >= 5.0
                break
