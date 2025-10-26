"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Return default configuration if file doesn't exist
        return get_default_config()
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            'num_samples': 10000,
            'noise_level': 0.1,
            'random_seed': 42
        },
        'models': {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5
        },
        'visualization': {
            'theme': 'plotly_white',
            'width': 800,
            'height': 600
        },
        'logging': {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
            'file': 'logs/fusion_analyzer.log'
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)