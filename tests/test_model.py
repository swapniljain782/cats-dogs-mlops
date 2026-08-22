"""Unit tests for model architecture and utility functions."""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import io
import json

from src.models.cnn_model import (
    create_cnn_model,
    compile_model,
    get_callbacks,
    save_model,
    load_model,
)


class TestCNNModel:
    """Tests for CNN model creation and compilation."""
    
    def test_create_cnn_model_default(self):
        """Test creating a CNN model with default parameters."""
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        
        assert model is not None
        assert model.output_shape == (None, 2)
        assert model.input_shape == (None, 224, 224, 3)
    
    def test_create_cnn_model_custom(self):
        """Test creating a CNN model with custom parameters."""
        model = create_cnn_model(input_shape=(128, 128, 3), num_classes=3)
        
        assert model.output_shape == (None, 3)
        assert model.input_shape == (None, 128, 128, 3)
    
    def test_compile_model(self):
        """Test compiling the model."""
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        compiled_model = compile_model(model, learning_rate=0.001)
        
        assert compiled_model.optimizer is not None
        assert compiled_model.loss is not None
        # Verify loss function name
        assert "sparse_categorical_crossentropy" in str(compiled_model.loss)
    
    def test_compile_model_custom_lr(self):
        """Test compiling with custom learning rate."""
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        compiled_model = compile_model(model, learning_rate=0.01)
        
        lr = float(compiled_model.optimizer.learning_rate)
        assert abs(lr - 0.01) < 1e-6
    
    def test_model_forward_pass(self):
        """Test model forward pass produces correct output shape."""
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        compiled_model = compile_model(model)
        
        # Create dummy input
        dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
        output = compiled_model.predict(dummy_input, verbose=0)
        
        assert output.shape == (2, 2)
        # Check probabilities sum to 1 (softmax)
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)
        # Check all probabilities are non-negative
        assert np.all(output >= 0)
    
    def test_get_callbacks(self):
        """Test callback generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model.keras")
            callbacks = get_callbacks(
                model_path=model_path,
                early_stopping_patience=5,
                reduce_lr_patience=3,
                reduce_lr_factor=0.5
            )
            
            assert len(callbacks) == 4
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert "EarlyStopping" in callback_types
            assert "ModelCheckpoint" in callback_types
            assert "ReduceLROnPlateau" in callback_types
            assert "CSVLogger" in callback_types
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        model = create_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        compiled_model = compile_model(model)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "test_model.keras")
            save_model(compiled_model, model_path)
            
            assert Path(model_path).exists()
            
            loaded_model = load_model(model_path)
            assert loaded_model is not None
            assert loaded_model.output_shape == compiled_model.output_shape
            
            # Verify predictions match
            dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            orig_pred = compiled_model.predict(dummy_input, verbose=0)
            loaded_pred = loaded_model.predict(dummy_input, verbose=0)
            np.testing.assert_array_almost_equal(orig_pred, loaded_pred, decimal=5)
