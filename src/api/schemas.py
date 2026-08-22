"""Pydantic schemas for API request/response validation."""
from typing import Dict, Optional
from pydantic import BaseModel, Field
import base64


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded image"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    model_config = {"protected_namespaces": ()}
    
    class_name: str = Field(..., description="Predicted class (cat or dog)")
    probability: float = Field(..., description="Confidence of predicted class")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for each class"
    )
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Loaded model version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None