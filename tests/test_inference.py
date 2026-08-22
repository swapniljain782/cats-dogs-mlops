"""Unit tests for model inference functions."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image

from src.api.model_loader import (
    preprocess_image,
    decode_base64_image,
    predict,
    predict_from_base64,
    load_model,
    get_model_version,
    get_class_names,
)
from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse


class TestModelLoader:
    """Tests for model loader functions."""
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test image
        img = Image.new("RGB", (300, 300), color="green")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        result = preprocess_image(img_bytes.getvalue(), target_size=(224, 224))
        
        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
    
    def test_decode_base64_image(self):
        """Test base64 decoding."""
        # Create test image and encode
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        import base64
        b64_string = base64.b64encode(img_bytes.getvalue()).decode()
        
        # Test with data URL prefix
        data_url = f"data:image/jpeg;base64,{b64_string}"
        decoded = decode_base64_image(data_url)
        assert decoded == img_bytes.getvalue()
        
        # Test without prefix
        decoded2 = decode_base64_image(b64_string)
        assert decoded2 == img_bytes.getvalue()
    
    @patch("src.api.model_loader.get_model")
    @patch("src.api.model_loader.get_class_names")
    def test_predict(self, mock_get_class_names, mock_get_model):
        """Test prediction function."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2, 0.8]])  # Dog with 0.8 confidence
        mock_get_model.return_value = mock_model
        mock_get_class_names.return_value = ["cat", "dog"]
        
        # Create test image
        img = Image.new("RGB", (224, 224), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        
        pred_class, pred_prob, class_probs = predict(img_bytes.getvalue())
        
        assert pred_class == "dog"
        assert pred_prob == 0.8
        assert class_probs == {"cat": 0.2, "dog": 0.8}
        mock_model.predict.assert_called_once()
    
    @patch("src.api.model_loader.predict")
    def test_predict_from_base64(self, mock_predict):
        """Test prediction from base64."""
        mock_predict.return_value = ("cat", 0.95, {"cat": 0.95, "dog": 0.05})
        
        # Use a valid base64 string (1x1 red JPEG)
        import base64
        img = Image.new("RGB", (1, 1), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        valid_b64 = base64.b64encode(buf.getvalue()).decode()
        
        pred_class, pred_prob, class_probs = predict_from_base64(valid_b64)
        
        assert pred_class == "cat"
        assert pred_prob == 0.95
        assert class_probs == {"cat": 0.95, "dog": 0.05}
        mock_predict.assert_called_once()


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_prediction_request(self):
        """Test PredictionRequest schema."""
        req = PredictionRequest(image_base64="test_base64")
        assert req.image_base64 == "test_base64"
    
    def test_prediction_response(self):
        """Test PredictionResponse schema."""
        resp = PredictionResponse(
            class_name="cat",
            probability=0.95,
            class_probabilities={"cat": 0.95, "dog": 0.05},
            model_version="1.0.0"
        )
        assert resp.class_name == "cat"
        assert resp.probability == 0.95
    
    def test_health_response(self):
        """Test HealthResponse schema."""
        resp = HealthResponse(
            status="healthy",
            model_version="1.0.0",
            model_loaded=True
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])