"""
Configuration Templates and Environment Management.

This module provides:
- Environment-specific configuration templates
- Configuration inheritance and overrides
- Dynamic environment switching
- Configuration deployment and rollback
- Configuration diff and merging utilities
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationTemplate:
    """Configuration template definition."""
    name: str
    description: str
    base_template: Optional[str] = None
    configuration: Dict[str, Any] = None
    required_overrides: List[str] = None
    created_at: datetime = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Initialize template."""
        if self.configuration is None:
            self.configuration = {}
        if self.required_overrides is None:
            self.required_overrides = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class EnvironmentConfiguration:
    """Environment-specific configuration."""
    environment: str
    template: str
    overrides: Dict[str, Any] = None
    feature_flags: Dict[str, bool] = None
    secrets: Dict[str, str] = None
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None
    
    def __post_init__(self):
        """Initialize environment configuration."""
        if self.overrides is None:
            self.overrides = {}
        if self.feature_flags is None:
            self.feature_flags = {}
        if self.secrets is None:
            self.secrets = {}


class ConfigurationTemplateManager:
    """Manages configuration templates and environments."""
    
    def __init__(self, templates_dir: str = "config/templates"):
        """
        Initialize template manager.
        
        Args:
            templates_dir: Directory containing configuration templates.
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates: Dict[str, ConfigurationTemplate] = {}
        self.environments: Dict[str, EnvironmentConfiguration] = {}
        
        self._load_templates()
        self._load_environments()
        
        logger.info(f"ConfigurationTemplateManager initialized with {len(self.templates)} templates")
    
    def create_template(self, template: ConfigurationTemplate) -> bool:
        """
        Create new configuration template.
        
        Args:
            template: Template configuration.
            
        Returns:
            True if successful.
        """
        try:
            self.templates[template.name] = template
            
            # Save to file
            template_file = self.templates_dir / f"{template.name}.yaml"
            template_data = asdict(template)
            template_data['created_at'] = template.created_at.isoformat()
            
            with open(template_file, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration template created: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating template {template.name}: {e}")
            return False
    
    def get_template(self, name: str) -> Optional[ConfigurationTemplate]:
        """
        Get configuration template.
        
        Args:
            name: Template name.
            
        Returns:
            Configuration template or None.
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """
        List available templates.
        
        Returns:
            List of template names.
        """
        return list(self.templates.keys())
    
    def resolve_template(self, template_name: str) -> Dict[str, Any]:
        """
        Resolve template with inheritance.
        
        Args:
            template_name: Template name to resolve.
            
        Returns:
            Resolved configuration dictionary.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        resolved_config = {}
        
        # Resolve base template first
        if template.base_template:
            resolved_config = self.resolve_template(template.base_template)
        
        # Merge current template configuration
        resolved_config = self._deep_merge(resolved_config, template.configuration)
        
        return resolved_config
    
    def create_environment(self, env_config: EnvironmentConfiguration) -> bool:
        """
        Create environment configuration.
        
        Args:
            env_config: Environment configuration.
            
        Returns:
            True if successful.
        """
        try:
            # Validate template exists
            if env_config.template not in self.templates:
                raise ValueError(f"Template not found: {env_config.template}")
            
            # Check required overrides
            template = self.templates[env_config.template]
            missing_overrides = set(template.required_overrides) - set(env_config.overrides.keys())
            if missing_overrides:
                raise ValueError(f"Missing required overrides: {missing_overrides}")
            
            self.environments[env_config.environment] = env_config
            
            # Save to file
            env_file = self.templates_dir / "environments" / f"{env_config.environment}.yaml"
            env_file.parent.mkdir(exist_ok=True)
            
            env_data = asdict(env_config)
            if env_config.deployed_at:
                env_data['deployed_at'] = env_config.deployed_at.isoformat()
            
            with open(env_file, 'w') as f:
                yaml.dump(env_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Environment configuration created: {env_config.environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating environment {env_config.environment}: {e}")
            return False
    
    def get_environment_configuration(self, environment: str) -> Dict[str, Any]:
        """
        Get complete environment configuration.
        
        Args:
            environment: Environment name.
            
        Returns:
            Complete configuration for environment.
        """
        if environment not in self.environments:
            raise ValueError(f"Environment not found: {environment}")
        
        env_config = self.environments[environment]
        
        # Resolve base template
        base_config = self.resolve_template(env_config.template)
        
        # Apply environment overrides
        final_config = self._deep_merge(base_config, env_config.overrides)
        
        # Add feature flags
        final_config['feature_flags'] = env_config.feature_flags
        
        # Add environment metadata
        final_config['_environment'] = {
            'name': environment,
            'template': env_config.template,
            'deployed_at': env_config.deployed_at.isoformat() if env_config.deployed_at else None,
            'deployed_by': env_config.deployed_by
        }
        
        return final_config
    
    def deploy_environment(self, environment: str, deployed_by: str) -> bool:
        """
        Deploy environment configuration.
        
        Args:
            environment: Environment to deploy.
            deployed_by: User deploying the configuration.
            
        Returns:
            True if successful.
        """
        try:
            if environment not in self.environments:
                raise ValueError(f"Environment not found: {environment}")
            
            # Update deployment metadata
            env_config = self.environments[environment]
            env_config.deployed_at = datetime.now()
            env_config.deployed_by = deployed_by
            
            # Save updated environment
            return self.create_environment(env_config)
            
        except Exception as e:
            logger.error(f"Error deploying environment {environment}: {e}")
            return False
    
    def compare_environments(self, env1: str, env2: str) -> Dict[str, Any]:
        """
        Compare two environment configurations.
        
        Args:
            env1: First environment name.
            env2: Second environment name.
            
        Returns:
            Configuration differences.
        """
        config1 = self.get_environment_configuration(env1)
        config2 = self.get_environment_configuration(env2)
        
        differences = self._compare_configurations(config1, config2, f"{env1} vs {env2}")
        
        return {
            "environment_1": env1,
            "environment_2": env2,
            "differences": differences,
            "total_differences": len(differences)
        }
    
    def validate_environment(self, environment: str) -> List[str]:
        """
        Validate environment configuration.
        
        Args:
            environment: Environment to validate.
            
        Returns:
            List of validation errors.
        """
        errors = []
        
        try:
            if environment not in self.environments:
                errors.append(f"Environment not found: {environment}")
                return errors
            
            env_config = self.environments[environment]
            
            # Check template exists
            if env_config.template not in self.templates:
                errors.append(f"Template not found: {env_config.template}")
                return errors
            
            # Check required overrides
            template = self.templates[env_config.template]
            missing_overrides = set(template.required_overrides) - set(env_config.overrides.keys())
            if missing_overrides:
                errors.append(f"Missing required overrides: {list(missing_overrides)}")
            
            # Try to resolve configuration
            try:
                self.get_environment_configuration(environment)
            except Exception as e:
                errors.append(f"Configuration resolution error: {e}")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def _load_templates(self):
        """Load templates from files."""
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                # Convert datetime
                if 'created_at' in template_data:
                    template_data['created_at'] = datetime.fromisoformat(template_data['created_at'])
                
                template = ConfigurationTemplate(**template_data)
                self.templates[template.name] = template
                
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    def _load_environments(self):
        """Load environment configurations from files."""
        env_dir = self.templates_dir / "environments"
        if not env_dir.exists():
            return
        
        for env_file in env_dir.glob("*.yaml"):
            try:
                with open(env_file, 'r') as f:
                    env_data = yaml.safe_load(f)
                
                # Convert datetime
                if 'deployed_at' in env_data and env_data['deployed_at']:
                    env_data['deployed_at'] = datetime.fromisoformat(env_data['deployed_at'])
                
                env_config = EnvironmentConfiguration(**env_data)
                self.environments[env_config.environment] = env_config
                
            except Exception as e:
                logger.error(f"Error loading environment {env_file}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _compare_configurations(self, config1: Dict[str, Any], config2: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """Compare two configurations and return differences."""
        differences = []
        
        # Find keys in config1 but not in config2
        for key in config1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config2:
                differences.append({
                    "type": "removed",
                    "path": current_path,
                    "value": config1[key]
                })
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                # Recursively compare nested dictionaries
                nested_diffs = self._compare_configurations(config1[key], config2[key], current_path)
                differences.extend(nested_diffs)
            elif config1[key] != config2[key]:
                differences.append({
                    "type": "changed",
                    "path": current_path,
                    "old_value": config1[key],
                    "new_value": config2[key]
                })
        
        # Find keys in config2 but not in config1
        for key in config2:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config1:
                differences.append({
                    "type": "added",
                    "path": current_path,
                    "value": config2[key]
                })
        
        return differences


def create_default_templates() -> List[ConfigurationTemplate]:
    """
    Create default configuration templates.
    
    Returns:
        List of default templates.
    """
    return [
        # Base template
        ConfigurationTemplate(
            name="base",
            description="Base configuration template for all environments",
            configuration={
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "handlers": ["console", "file"]
                },
                "database": {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_timeout": 30
                },
                "api": {
                    "rate_limit": 1000,
                    "timeout": 30,
                    "max_request_size": "10MB"
                },
                "fusion": {
                    "safety_threshold": 1e21,
                    "temperature_limit": 200e6,
                    "density_limit": 1e22
                },
                "ml": {
                    "model_cache_size": 100,
                    "prediction_timeout": 30,
                    "feature_selection": True
                }
            }
        ),
        
        # Development template
        ConfigurationTemplate(
            name="development",
            description="Development environment configuration",
            base_template="base",
            configuration={
                "logging": {
                    "level": "DEBUG"
                },
                "database": {
                    "url": "sqlite:///fusion_dev.db",
                    "echo": True
                },
                "api": {
                    "debug": True,
                    "rate_limit": 10000
                },
                "fusion": {
                    "enable_simulation": True,
                    "mock_sensors": True
                },
                "ml": {
                    "use_gpu": False,
                    "model_validation": True
                }
            },
            required_overrides=[]
        ),
        
        # Production template
        ConfigurationTemplate(
            name="production",
            description="Production environment configuration",
            base_template="base",
            configuration={
                "logging": {
                    "level": "WARNING",
                    "handlers": ["file", "syslog"]
                },
                "database": {
                    "pool_size": 50,
                    "max_overflow": 100,
                    "echo": False
                },
                "api": {
                    "debug": False,
                    "rate_limit": 1000,
                    "enable_cors": False
                },
                "fusion": {
                    "enable_simulation": False,
                    "mock_sensors": False,
                    "safety_monitoring": True
                },
                "ml": {
                    "use_gpu": True,
                    "model_validation": False,
                    "batch_prediction": True
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_interval": 60,
                    "alert_thresholds": {
                        "cpu_usage": 80,
                        "memory_usage": 85,
                        "error_rate": 5
                    }
                }
            },
            required_overrides=["database.url", "api.secret_key", "monitoring.endpoints"]
        ),
        
        # Testing template
        ConfigurationTemplate(
            name="testing",
            description="Testing environment configuration",
            base_template="base",
            configuration={
                "logging": {
                    "level": "ERROR"
                },
                "database": {
                    "url": "sqlite:///:memory:",
                    "pool_size": 1
                },
                "api": {
                    "testing": True,
                    "rate_limit": 100000
                },
                "fusion": {
                    "enable_simulation": True,
                    "mock_sensors": True,
                    "deterministic_random": True
                },
                "ml": {
                    "use_gpu": False,
                    "model_cache_size": 10,
                    "fast_training": True
                }
            }
        ),
        
        # Staging template
        ConfigurationTemplate(
            name="staging",
            description="Staging environment configuration",
            base_template="production",
            configuration={
                "logging": {
                    "level": "INFO"
                },
                "api": {
                    "debug": True,
                    "rate_limit": 5000
                },
                "fusion": {
                    "enable_simulation": True
                },
                "monitoring": {
                    "metrics_interval": 30
                }
            },
            required_overrides=["database.url", "api.secret_key"]
        )
    ]


def create_default_environments() -> List[EnvironmentConfiguration]:
    """
    Create default environment configurations.
    
    Returns:
        List of default environments.
    """
    return [
        EnvironmentConfiguration(
            environment="development",
            template="development",
            feature_flags={
                "advanced_ml_models": True,
                "real_time_monitoring": False,
                "experimental_algorithms": True,
                "enhanced_security": False
            }
        ),
        
        EnvironmentConfiguration(
            environment="testing",
            template="testing",
            feature_flags={
                "advanced_ml_models": True,
                "real_time_monitoring": False,
                "experimental_algorithms": False,
                "enhanced_security": True
            }
        ),
        
        EnvironmentConfiguration(
            environment="staging",
            template="staging",
            overrides={
                "database": {
                    "url": "postgresql://user:pass@staging-db:5432/fusion"
                },
                "api": {
                    "secret_key": "staging-secret-key"
                }
            },
            feature_flags={
                "advanced_ml_models": True,
                "real_time_monitoring": True,
                "experimental_algorithms": False,
                "enhanced_security": True
            }
        ),
        
        EnvironmentConfiguration(
            environment="production",
            template="production",
            overrides={
                "database": {
                    "url": "postgresql://user:pass@prod-db:5432/fusion"
                },
                "api": {
                    "secret_key": "production-secret-key"
                },
                "monitoring": {
                    "endpoints": ["http://prometheus:9090", "http://grafana:3000"]
                }
            },
            feature_flags={
                "advanced_ml_models": True,
                "real_time_monitoring": True,
                "experimental_algorithms": False,
                "enhanced_security": True
            }
        )
    ]


def setup_default_configuration(templates_dir: str = "config/templates") -> ConfigurationTemplateManager:
    """
    Setup default configuration templates and environments.
    
    Args:
        templates_dir: Templates directory.
        
    Returns:
        Configured template manager.
    """
    manager = ConfigurationTemplateManager(templates_dir)
    
    # Create default templates
    for template in create_default_templates():
        manager.create_template(template)
    
    # Create default environments
    for env_config in create_default_environments():
        manager.create_environment(env_config)
    
    logger.info("Default configuration templates and environments created")
    return manager