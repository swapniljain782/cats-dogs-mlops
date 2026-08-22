"""Model loading and inference utilities."""
import os
import json
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image
import tensorflow as tf
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global model cache
_model = None
_model_version = "unknown"
_class_names = ["cat", "dog"]


def load_model(model_path: str = None) -> tf.keras.Model:
    """Load the trained model."""
    global _model, _model_version, _class_names
    
    if model_path is None:
        model_path = "models/best_model/model.keras"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    _model = tf.keras.models.load_model(model_path)
    
    # Try to load metadata
    meta_path = Path(model_path).parent / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            _class_names = metadata.get("class_names", ["cat", "dog"])
            _model_version = metadata.get("model_version", "1.0.0")
    else:
        _model_version = "1.0.0"
    
    logger.info(f"Model loaded. Version: {_model_version}, Classes: {_class_names}")
    return _model


def get_model() -> tf.keras.Model:
    """Get the loaded model, loading if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


def get_model_version() -> str:
    """Get the model version."""
    global _model_version
    if _model_version == "unknown":
        load_model()
    return _model_version


def get_class_names() -> list:
    """Get the class names."""
    global _class_names
    if _class_names == ["cat", "dog"]:
        load_model()
    return _class_names


def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image bytes for model input."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    return base64.b64decode(base64_string)


def predict(image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
    """Run inference on image bytes."""
    model = get_model()
    class_names = get_class_names()
    config = get_config()
    image_size = (config.data.image_size, config.data.image_size)
    
    # Preprocess
    img_array = preprocess_image(image_bytes, target_size=image_size)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class
    pred_idx = int(np.argmax(probabilities))
    pred_class = class_names[pred_idx]
    pred_prob = float(probabilities[pred_idx])
    
    # Format probabilities
    class_probs = {name: float(prob) for name, prob in zip(class_names, probabilities)}
    
    return pred_class, pred_prob, class_probs


def predict_from_base64(base64_string: str) -> Tuple[str, float, Dict[str, float]]:
    """Run inference on base64 encoded image."""
    image_bytes = decode_base64_image(base64_string)
    return predict(image_bytes)