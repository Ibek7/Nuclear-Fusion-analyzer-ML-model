"""
Configuration management system for Nuclear Fusion Analyzer.

This module provides utilities for loading and managing configuration files
using Hydra for hierarchical configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class ConfigPaths:
    """Default paths for configuration files."""
    CONFIG_DIR = Path(__file__).parent
    DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
    DEV_CONFIG = CONFIG_DIR / "development.yaml"
    PROD_CONFIG = CONFIG_DIR / "production.yaml"


class ConfigManager:
    """
    Configuration manager for the Nuclear Fusion Analyzer.
    
    Handles loading, merging, and accessing configuration parameters
    from YAML files with environment-specific overrides.
    """
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_name: Name of the configuration file (without .yaml extension)
            config_path: Custom path to configuration directory
        """
        self.config_name = config_name
        self.config_path = Path(config_path) if config_path else ConfigPaths.CONFIG_DIR
        self.config: Optional[DictConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        config_file = self.config_path / f"{self.config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to OmegaConf for dot notation access
            self.config = OmegaConf.create(config_dict)
            
            # Handle inheritance (defaults key)
            if 'defaults' in self.config:
                self._resolve_defaults()
                
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _resolve_defaults(self) -> None:
        """Resolve configuration inheritance from defaults."""
        if 'defaults' not in self.config:
            return
            
        defaults = self.config.defaults
        if isinstance(defaults, (list, tuple)):
            # Load and merge default configurations
            for default_name in defaults:
                if isinstance(default_name, str):
                    default_config = self._load_default_config(default_name)
                    # Merge with current config (current takes precedence)
                    self.config = OmegaConf.merge(default_config, self.config)
        
        # Remove defaults key from final config
        if 'defaults' in self.config:
            del self.config['defaults']
    
    def _load_default_config(self, config_name: str) -> DictConfig:
        """Load a default configuration file."""
        default_file = self.config_path / f"{config_name}.yaml"
        
        if not default_file.exists():
            raise FileNotFoundError(f"Default configuration not found: {default_file}")
        
        with open(default_file, 'r') as f:
            default_dict = yaml.safe_load(f)
        
        return OmegaConf.create(default_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'models.random_forest.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self.config is None:
            return default
            
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'models', 'data')
            
        Returns:
            Dictionary containing section configuration
        """
        if self.config is None:
            return {}
            
        section_config = self.get(section, {})
        if isinstance(section_config, DictConfig):
            return OmegaConf.to_container(section_config, resolve=True)
        return section_config
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates in dot notation
        """
        if self.config is None:
            return
            
        for key, value in updates.items():
            OmegaConf.set(self.config, key, value)
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
            
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if self.config is None:
            return {}
            
        return OmegaConf.to_container(self.config, resolve=True)
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if self.config is None:
            validation_results['valid'] = False
            validation_results['errors'].append("No configuration loaded")
            return validation_results
        
        # Validate data section
        self._validate_data_config(validation_results)
        
        # Validate models section
        self._validate_models_config(validation_results)
        
        # Validate anomaly detection section
        self._validate_anomaly_config(validation_results)
        
        # Set overall validity
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    def _validate_data_config(self, results: Dict[str, Any]) -> None:
        """Validate data configuration section."""
        data_config = self.get_section('data')
        
        if 'default_samples' in data_config:
            if data_config['default_samples'] <= 0:
                results['errors'].append("default_samples must be positive")
        
        if 'test_size' in data_config:
            test_size = data_config['test_size']
            if not (0 < test_size < 1):
                results['errors'].append("test_size must be between 0 and 1")
        
        if 'validation_size' in data_config:
            val_size = data_config['validation_size']
            if not (0 < val_size < 1):
                results['errors'].append("validation_size must be between 0 and 1")
    
    def _validate_models_config(self, results: Dict[str, Any]) -> None:
        """Validate models configuration section."""
        models_config = self.get_section('models')
        
        # Validate random forest parameters
        if 'random_forest' in models_config:
            rf_config = models_config['random_forest']
            if 'n_estimators' in rf_config and rf_config['n_estimators'] <= 0:
                results['errors'].append("random_forest.n_estimators must be positive")
        
        # Validate deep learning parameters
        if 'deep_learning' in models_config:
            dl_config = models_config['deep_learning']
            if 'epochs' in dl_config and dl_config['epochs'] <= 0:
                results['errors'].append("deep_learning.epochs must be positive")
    
    def _validate_anomaly_config(self, results: Dict[str, Any]) -> None:
        """Validate anomaly detection configuration section."""
        anomaly_config = self.get_section('anomaly_detection')
        
        if 'isolation_forest' in anomaly_config:
            if_config = anomaly_config['isolation_forest']
            contamination = if_config.get('contamination', 0.1)
            if not (0 < contamination < 0.5):
                results['warnings'].append("isolation_forest.contamination should be between 0 and 0.5")


def load_config(config_name: str = "default", config_path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration using the specified config name.
    
    Args:
        config_name: Name of configuration file (without .yaml extension)
        config_path: Custom path to configuration directory
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_name, config_path)


def get_config_for_environment(environment: str = "development") -> ConfigManager:
    """
    Get configuration for specific environment.
    
    Args:
        environment: Environment name ('development', 'production', 'default')
        
    Returns:
        ConfigManager instance for the environment
    """
    env_mapping = {
        'dev': 'development',
        'development': 'development',
        'prod': 'production', 
        'production': 'production',
        'default': 'default'
    }
    
    config_name = env_mapping.get(environment.lower(), 'default')
    return load_config(config_name)


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_global_config() -> ConfigManager:
    """Get or create global configuration instance."""
    global _global_config
    if _global_config is None:
        # Try to determine environment from environment variable
        env = os.getenv('FUSION_ENV', 'development')
        _global_config = get_config_for_environment(env)
    return _global_config


def set_global_config(config: ConfigManager) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config


# Convenience functions for common configuration access
def get_data_config() -> Dict[str, Any]:
    """Get data configuration section."""
    return get_global_config().get_section('data')


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for specific model."""
    return get_global_config().get_section(f'models.{model_name}')


def get_anomaly_config() -> Dict[str, Any]:
    """Get anomaly detection configuration section."""
    return get_global_config().get_section('anomaly_detection')


def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration section."""
    return get_global_config().get_section('visualization')


def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation configuration section."""
    return get_global_config().get_section('evaluation')