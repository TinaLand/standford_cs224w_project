"""
Configuration Management Utilities
==================================

Centralized configuration loading from YAML file.
All scripts should use this module to load configuration instead of hardcoding values.

Usage:
    from utils_config import load_config, get_config
    
    config = load_config()
    data_dir = get_config('paths.data_raw')
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


CONFIG_FILE = Path(__file__).resolve().parent.parent / 'config.yaml'
_config_cache: Optional[Dict] = None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: project_root/config.yaml)
    
    Returns:
        Dictionary with configuration values
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if config_path is None:
        config_path = CONFIG_FILE
    
    if not config_path.exists():
        print(f"⚠️  Warning: Config file not found at {config_path}")
        print("   Using default configuration. Create config.yaml for customization.")
        return _get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        _config_cache = config
        return config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("   Using default configuration.")
        return _get_default_config()


def get_config(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Examples:
        get_config('data.start_date') -> '2015-01-01'
        get_config('baseline.learning_rate') -> 0.001
        get_config('nonexistent.key', default='default_value') -> 'default_value'
    
    Args:
        key_path: Dot-separated path to config value (e.g., 'data.start_date')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    config = load_config()
    
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config(key_path: str, value: Any) -> None:
    """
    Set a configuration value (updates cache only, doesn't save to file).
    
    Args:
        key_path: Dot-separated path to config value
        value: Value to set
    """
    global _config_cache
    
    config = load_config()
    keys = key_path.split('.')
    
    # Navigate to parent dict
    target = config
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    
    # Set value
    target[keys[-1]] = value
    _config_cache = config


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration if config file doesn't exist."""
    return {
        'data': {
            'start_date': '2015-01-01',
            'end_date': '2025-01-01',
            'stock_source': 'SPY',
            'num_stocks': 50
        },
        'paths': {
            'project_root': '.',
            'data_raw': 'data/raw',
            'data_processed': 'data/processed',
            'data_graphs': 'data/graphs',
            'models': 'models'
        },
        'reproducibility': {
            'random_seed': 42,
            'pytorch_deterministic': True
        }
    }


def setup_reproducibility(config: Optional[Dict] = None) -> None:
    """
    Set up random seeds and deterministic flags for reproducibility.
    
    Args:
        config: Configuration dict (if None, loads from file)
    """
    if config is None:
        config = load_config()
    
    repro_config = config.get('reproducibility', {})
    seed = repro_config.get('random_seed', 42)
    pytorch_deterministic = repro_config.get('pytorch_deterministic', True)
    cudnn_deterministic = repro_config.get('cudnn_deterministic', True)
    cudnn_benchmark = repro_config.get('cudnn_benchmark', False)
    
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark
        
        if pytorch_deterministic:
            # Note: This may reduce performance
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        print(f"✅ Reproducibility set up: seed={seed}, deterministic={pytorch_deterministic}")
    except ImportError:
        print("⚠️  PyTorch not available, skipping PyTorch seed setup")


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Data start date: {get_config('data.start_date')}")
    print(f"Baseline learning rate: {get_config('baseline.learning_rate')}")

