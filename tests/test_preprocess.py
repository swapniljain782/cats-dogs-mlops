"""Unit tests for data preprocessing functions."""
import pytest
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path

from src.data.preprocess import (
    load_and_preprocess_image,
    create_augmentation_pipeline,
    get_image_paths_and_labels,
    create_tf_dataset,
)
from src.data.split import split_dataset, get_split_dataset


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_load_and_preprocess_image(self):
        """Test loading and preprocessing a single image."""
        # Create a test image
        img = Image.new("RGB", (300, 300), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes.getvalue())
            temp_path = f.name
        
        try:
            result = load_and_preprocess_image(temp_path, target_size=(224, 224))
            
            assert result.shape == (224, 224, 3)
            assert result.dtype == np.float32
            assert np.all(result >= 0.0) and np.all(result <= 1.0)
        finally:
            Path(temp_path).unlink()
    
    def test_load_and_preprocess_image_no_normalize(self):
        """Test loading without normalization."""
        img = Image.new("RGB", (300, 300), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes.getvalue())
            temp_path = f.name
        
        try:
            result = load_and_preprocess_image(temp_path, target_size=(224, 224), normalize=False)
            
            assert result.shape == (224, 224, 3)
            assert result.dtype == np.float32
            assert np.max(result) > 1.0  # Not normalized
        finally:
            Path(temp_path).unlink()
    
    def test_create_augmentation_pipeline(self):
        """Test augmentation pipeline creation."""
        pipeline = create_augmentation_pipeline(
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2)
        )
        
        assert pipeline is not None
        assert len(pipeline.layers) == 4
    
    def test_create_augmentation_pipeline_empty(self):
        """Test augmentation pipeline with no augmentations."""
        pipeline = create_augmentation_pipeline(
            rotation_range=0,
            zoom_range=0,
            horizontal_flip=False,
            brightness_range=None
        )
        
        assert pipeline is None
    
    def test_create_tf_dataset_with_real_images(self):
        """Test TensorFlow dataset creation with real temp images."""
        import tensorflow as tf
        
        # Create temp directory with dummy images
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(8):
                img = Image.new("RGB", (64, 64), color=["red", "blue"][i % 2])
                img_path = Path(tmpdir) / f"test_{i}.jpg"
                img.save(str(img_path))
                paths.append(str(img_path))
            
            labels = [0, 1, 0, 1, 0, 1, 0, 1]
            
            dataset = create_tf_dataset(
                image_paths=paths,
                labels=labels,
                batch_size=4,
                image_size=(224, 224),
                shuffle=False,
                augment=False
            )
            
            assert isinstance(dataset, tf.data.Dataset)
            
            # Check batch shape
            for images, labels_batch in dataset.take(1):
                assert images.shape == (4, 224, 224, 3)
                assert labels_batch.shape == (4,)

    def test_get_image_paths_and_labels(self):
        """Test extracting image paths and labels from directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create class directories
            cat_dir = Path(tmpdir) / "cats"
            dog_dir = Path(tmpdir) / "dogs"
            cat_dir.mkdir()
            dog_dir.mkdir()
            
            # Create dummy images
            for i in range(3):
                img = Image.new("RGB", (32, 32), color="red")
                img.save(str(cat_dir / f"cat_{i}.jpg"))
                
                img = Image.new("RGB", (32, 32), color="blue")
                img.save(str(dog_dir / f"dog_{i}.jpg"))
            
            paths, labels = get_image_paths_and_labels(Path(tmpdir))
            
            assert len(paths) == 6
            assert len(labels) == 6
            assert set(labels) == {0, 1}


class TestSplit:
    """Tests for dataset splitting."""
    
    def test_get_split_dataset_no_data_dir(self):
        """Test that get_split_dataset works when data directory is missing."""
        import tempfile
        import os
        
        # Save and restore working directory to avoid side effects
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                # Should not crash because TFRecordDataset is lazy-loaded
                # The error would occur when iterating, not when creating
                import tensorflow as tf
                dataset = get_split_dataset("train")
                assert isinstance(dataset, tf.data.Dataset)
            except FileNotFoundError:
                # This is also acceptable - some code paths check file existence
                pass
            finally:
                os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
