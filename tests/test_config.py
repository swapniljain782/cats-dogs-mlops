"""Unit tests for configuration and utilities."""
import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils.config import load_config, get_config, get_config_obj, Config


class TestConfig:
    """Tests for configuration management."""
    
    def test_load_config(self):
        """Test loading config from params.yaml."""
        config = load_config()
        assert config is not None
        assert "data" in config
        assert "model" in config
        assert "mlflow" in config
    
    def test_load_config_custom_path(self):
        """Test loading config from custom path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"test": {"value": 42}}, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config["test"]["value"] == 42
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_not_found(self):
        """Test loading config from non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
    
    def test_get_config_object(self):
        """Test get_config() returns Config object when called with no args."""
        config = get_config()
        assert isinstance(config, Config)
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
    
    def test_get_config_value(self):
        """Test get_config() with dot-notation key returns value."""
        batch_size = get_config("data.batch_size")
        assert batch_size == 64
    
    def test_get_config_value_with_default(self):
        """Test get_config() with default for missing key."""
        value = get_config("nonexistent.key", "default_value")
        assert value == "default_value"
    
    def test_get_config_nested(self):
        """Test get_config() with deeply nested key."""
        lr = get_config("model.learning_rate")
        assert lr == 0.001
    
    def test_config_attribute_access(self):
        """Test Config attribute access."""
        config = get_config_obj()
        assert config.data.image_size == 224
        assert config.model.epochs == 20
        assert config.data.batch_size == 64
    
    def test_config_dot_get(self):
        """Test Config.get() method with dot notation."""
        config = get_config_obj()
        assert config.get("data.image_size") == 224
        assert config.get("nonexistent.key", "default") == "default"
    
    def test_config_to_dict(self):
        """Test Config.to_dict() conversion."""
        config = get_config_obj()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "data" in d
        assert "model" in d
    
    def test_config_immutable_cache(self):
        """Test that load_config is cached."""
        config1 = load_config()
        config2 = load_config()
        assert config1 is config2  # Same object due to lru_cache
