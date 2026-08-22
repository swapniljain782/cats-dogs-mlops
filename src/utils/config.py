"""Configuration management using YAML files."""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Configuration class with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation (e.g., 'data.batch_size')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config


@lru_cache(maxsize=1)
def get_config_obj() -> Config:
    """Get configuration as an object with attribute access."""
    return Config(load_config())


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """Get configuration value or Config object.
    
    - get_config() → returns Config object with attribute access
    - get_config("data.batch_size") → returns the specific value
    - get_config("data.batch_size", 32) → returns value with default fallback
    """
    if key is None:
        return get_config_obj()
    
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default
    return value
