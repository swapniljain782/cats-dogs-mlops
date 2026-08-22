"""API modules for the cats-dogs MLOps pipeline."""
from src.api.main import app
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.model_loader import (
    load_model,
    get_model,
    get_model_version,
    predict,
    predict_from_base64
)

__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "ErrorResponse",
    "load_model",
    "get_model",
    "get_model_version",
    "predict",
    "predict_from_base64",
]