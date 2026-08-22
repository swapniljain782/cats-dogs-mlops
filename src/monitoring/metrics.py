"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from src.utils.config import get_config

# HTTP metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Model metrics
model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["predicted_class"]
)

model_prediction_latency_seconds = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

model_prediction_confidence = Histogram(
    "model_prediction_confidence",
    "Model prediction confidence",
    ["predicted_class"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# System metrics
model_loaded = Gauge(
    "model_loaded",
    "Whether model is loaded (1) or not (0)"
)

model_version_info = Gauge(
    "model_version_info",
    "Model version info",
    ["version"]
)


def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def record_prediction(predicted_class: str, latency: float, confidence: float):
    """Record model prediction metrics."""
    model_predictions_total.labels(predicted_class=predicted_class).inc()
    model_prediction_latency_seconds.observe(latency)
    model_prediction_confidence.labels(predicted_class=predicted_class).observe(confidence)


def set_model_loaded(loaded: bool, version: str = "unknown"):
    """Set model loaded status and version."""
    model_loaded.set(1 if loaded else 0)
    if loaded:
        model_version_info.labels(version=version).set(1)


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format."""
    return generate_latest()