"""
Advanced Configuration Management System.

This module provides:
- Environment-specific configuration management
- Dynamic configuration loading and hot-reloading
- Feature flags and toggles
- Configuration validation and schema enforcement
- Secrets management and encryption
- Configuration versioning and rollback
- Distributed configuration management
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import base64

# Cryptography imports (optional)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    Fernet = None

logger = logging.getLogger(__name__)


class ConfigurationSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


class ConfigurationType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"


@dataclass
class ConfigurationSchema:
    """Schema definition for configuration values."""
    name: str
    type: ConfigurationType
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate configuration value against schema.
        
        Args:
            value: Value to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check if value is None and required
        if value is None:
            if self.required:
                return False, f"Required configuration '{self.name}' is missing"
            return True, None
        
        # Type validation
        if self.type == ConfigurationType.STRING:
            if not isinstance(value, str):
                return False, f"Configuration '{self.name}' must be a string"
        elif self.type == ConfigurationType.INTEGER:
            if not isinstance(value, int):
                return False, f"Configuration '{self.name}' must be an integer"
        elif self.type == ConfigurationType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Configuration '{self.name}' must be a float"
        elif self.type == ConfigurationType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Configuration '{self.name}' must be a boolean"
        elif self.type == ConfigurationType.LIST:
            if not isinstance(value, list):
                return False, f"Configuration '{self.name}' must be a list"
        elif self.type == ConfigurationType.DICT:
            if not isinstance(value, dict):
                return False, f"Configuration '{self.name}' must be a dictionary"
        
        # Range validation for numeric types
        if self.type in [ConfigurationType.INTEGER, ConfigurationType.FLOAT]:
            if self.min_value is not None and value < self.min_value:
                return False, f"Configuration '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Configuration '{self.name}' must be <= {self.max_value}"
        
        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Configuration '{self.name}' must be one of {self.allowed_values}"
        
        # Pattern validation for strings
        if self.type == ConfigurationType.STRING and self.pattern:
            import re
            if not re.match(self.pattern, value):
                return False, f"Configuration '{self.name}' does not match pattern {self.pattern}"
        
        return True, None


@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    enabled: bool
    description: Optional[str] = None
    rollout_percentage: float = 100.0
    user_groups: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dependencies: Optional[List[str]] = None
    
    def is_enabled_for_user(self, user_id: str, user_groups: Optional[List[str]] = None) -> bool:
        """
        Check if feature is enabled for specific user.
        
        Args:
            user_id: User identifier.
            user_groups: User's groups.
            
        Returns:
            True if feature is enabled for user.
        """
        if not self.enabled:
            return False
        
        # Check time-based activation
        now = datetime.now()
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        
        # Check user groups
        if self.user_groups and user_groups:
            if not any(group in self.user_groups for group in user_groups):
                return False
        
        # Check rollout percentage
        if self.rollout_percentage < 100.0:
            # Use deterministic hash for consistent user experience
            user_hash = hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest()
            user_percentage = int(user_hash[:8], 16) % 100
            return user_percentage < self.rollout_percentage
        
        return True


@dataclass
class ConfigurationValue:
    """Represents a configuration value with metadata."""
    key: str
    value: Any
    source: ConfigurationSource
    encrypted: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value if not self.encrypted else "***ENCRYPTED***",
            "source": self.source.value,
            "encrypted": self.encrypted,
            "last_updated": self.last_updated.isoformat(),
            "version": self.version
        }


class ConfigurationProvider(ABC):
    """Abstract base class for configuration providers."""
    
    @abstractmethod
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass
    
    @abstractmethod
    def save_configuration(self, config: Dict[str, Any]) -> bool:
        """Save configuration to source."""
        pass
    
    @abstractmethod
    def watch_changes(self, callback: Callable[[Dict[str, Any]], None]):
        """Watch for configuration changes."""
        pass


class FileConfigurationProvider(ConfigurationProvider):
    """File-based configuration provider."""
    
    def __init__(self, file_path: str, file_format: str = "auto"):
        """
        Initialize file configuration provider.
        
        Args:
            file_path: Path to configuration file.
            file_format: File format (yaml, json, auto).
        """
        self.file_path = Path(file_path)
        self.file_format = file_format
        self.last_modified = None
        self.watchers = []
        
        if file_format == "auto":
            self.file_format = "yaml" if self.file_path.suffix in [".yml", ".yaml"] else "json"
        
        logger.info(f"FileConfigurationProvider initialized: {file_path}")
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            logger.warning(f"Configuration file not found: {self.file_path}")
            return {}
        
        try:
            with open(self.file_path, 'r') as f:
                if self.file_format == "yaml":
                    config = yaml.safe_load(f) or {}
                else:
                    config = json.load(f)
            
            self.last_modified = self.file_path.stat().st_mtime
            logger.info(f"Configuration loaded from {self.file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {self.file_path}: {e}")
            return {}
    
    def save_configuration(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'w') as f:
                if self.file_format == "yaml":
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config, f, indent=2, default=str)
            
            self.last_modified = self.file_path.stat().st_mtime
            logger.info(f"Configuration saved to {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {self.file_path}: {e}")
            return False
    
    def watch_changes(self, callback: Callable[[Dict[str, Any]], None]):
        """Watch for file changes."""
        self.watchers.append(callback)
        
        def watch_thread():
            while True:
                try:
                    if self.file_path.exists():
                        current_modified = self.file_path.stat().st_mtime
                        if self.last_modified and current_modified > self.last_modified:
                            config = self.load_configuration()
                            for watcher in self.watchers:
                                try:
                                    watcher(config)
                                except Exception as e:
                                    logger.error(f"Error in configuration watcher: {e}")
                    
                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    logger.error(f"Error watching configuration file: {e}")
                    time.sleep(5.0)
        
        threading.Thread(target=watch_thread, daemon=True).start()


class EnvironmentConfigurationProvider(ConfigurationProvider):
    """Environment variable configuration provider."""
    
    def __init__(self, prefix: str = "FUSION_"):
        """
        Initialize environment configuration provider.
        
        Args:
            prefix: Environment variable prefix.
        """
        self.prefix = prefix
        logger.info(f"EnvironmentConfigurationProvider initialized with prefix: {prefix}")
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()
                
                # Try to parse as JSON first, then as string
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # Try boolean conversion
                    if value.lower() in ['true', 'false']:
                        config[config_key] = value.lower() == 'true'
                    # Try numeric conversion
                    elif value.isdigit():
                        config[config_key] = int(value)
                    elif value.replace('.', '').isdigit():
                        config[config_key] = float(value)
                    else:
                        config[config_key] = value
        
        logger.info(f"Loaded {len(config)} configuration values from environment")
        return config
    
    def save_configuration(self, config: Dict[str, Any]) -> bool:
        """Save configuration to environment (not persistent)."""
        try:
            for key, value in config.items():
                env_key = f"{self.prefix}{key.upper()}"
                os.environ[env_key] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to environment: {e}")
            return False
    
    def watch_changes(self, callback: Callable[[Dict[str, Any]], None]):
        """Environment variables don't support watching."""
        pass


class SecretManager:
    """Manages encrypted secrets in configuration."""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize secret manager.
        
        Args:
            master_key: Master encryption key.
        """
        if not HAS_CRYPTO:
            logger.warning("Cryptography library not available - secrets will not be encrypted")
            self.cipher = None
            return
        
        if master_key is None:
            master_key = os.getenv("FUSION_MASTER_KEY")
        
        if master_key:
            # Derive key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'fusion_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            self.cipher = Fernet(key)
        else:
            logger.warning("No master key provided - secrets will not be encrypted")
            self.cipher = None
    
    def encrypt_secret(self, value: str) -> str:
        """
        Encrypt a secret value.
        
        Args:
            value: Value to encrypt.
            
        Returns:
            Encrypted value.
        """
        if not self.cipher:
            logger.warning("No cipher available - returning unencrypted value")
            return value
        
        try:
            encrypted = self.cipher.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Error encrypting secret: {e}")
            return value
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """
        Decrypt a secret value.
        
        Args:
            encrypted_value: Encrypted value to decrypt.
            
        Returns:
            Decrypted value.
        """
        if not self.cipher:
            logger.warning("No cipher available - returning encrypted value")
            return encrypted_value
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Error decrypting secret: {e}")
            return encrypted_value


class ConfigurationManager:
    """Advanced configuration management system."""
    
    def __init__(self, 
                 environment: str = "development",
                 providers: Optional[List[ConfigurationProvider]] = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Current environment.
            providers: List of configuration providers.
        """
        self.environment = environment
        self.providers = providers or []
        self.configuration: Dict[str, ConfigurationValue] = {}
        self.schemas: Dict[str, ConfigurationSchema] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.secret_manager = SecretManager()
        self.lock = threading.RLock()
        self.change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # Default providers
        if not self.providers:
            self.providers = [
                EnvironmentConfigurationProvider(),
                FileConfigurationProvider(f"config/{environment}.yaml")
            ]
        
        logger.info(f"ConfigurationManager initialized for environment: {environment}")
        self._load_all_configurations()
        self._setup_watchers()
    
    def add_provider(self, provider: ConfigurationProvider):
        """
        Add configuration provider.
        
        Args:
            provider: Configuration provider to add.
        """
        with self.lock:
            self.providers.append(provider)
            self._load_provider_configuration(provider)
    
    def add_schema(self, schema: ConfigurationSchema):
        """
        Add configuration schema for validation.
        
        Args:
            schema: Configuration schema.
        """
        with self.lock:
            self.schemas[schema.name] = schema
            
            # Validate existing configuration against new schema
            if schema.name in self.configuration:
                value = self.configuration[schema.name].value
                is_valid, error = schema.validate(value)
                if not is_valid:
                    logger.warning(f"Existing configuration violates schema: {error}")
    
    def add_feature_flag(self, feature_flag: FeatureFlag):
        """
        Add feature flag.
        
        Args:
            feature_flag: Feature flag configuration.
        """
        with self.lock:
            self.feature_flags[feature_flag.name] = feature_flag
        
        logger.info(f"Feature flag added: {feature_flag.name} (enabled: {feature_flag.enabled})")
    
    def get(self, key: str, default: Any = None, decrypt: bool = True) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key not found.
            decrypt: Whether to decrypt secret values.
            
        Returns:
            Configuration value.
        """
        with self.lock:
            if key not in self.configuration:
                # Check if there's a schema with a default value
                if key in self.schemas and self.schemas[key].default is not None:
                    return self.schemas[key].default
                return default
            
            config_value = self.configuration[key]
            
            if config_value.encrypted and decrypt:
                return self.secret_manager.decrypt_secret(config_value.value)
            
            return config_value.value
    
    def set(self, key: str, value: Any, source: ConfigurationSource = ConfigurationSource.MEMORY, 
            encrypt: bool = False) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
            source: Configuration source.
            encrypt: Whether to encrypt the value.
            
        Returns:
            True if successful.
        """
        with self.lock:
            # Validate against schema if available
            if key in self.schemas:
                is_valid, error = self.schemas[key].validate(value)
                if not is_valid:
                    logger.error(f"Configuration validation failed: {error}")
                    return False
            
            # Encrypt if requested
            actual_value = value
            if encrypt and isinstance(value, str):
                actual_value = self.secret_manager.encrypt_secret(value)
            
            # Get old value for change notification
            old_value = self.configuration[key].value if key in self.configuration else None
            
            # Update configuration
            if key in self.configuration:
                self.configuration[key].value = actual_value
                self.configuration[key].encrypted = encrypt
                self.configuration[key].last_updated = datetime.now()
                self.configuration[key].version += 1
            else:
                self.configuration[key] = ConfigurationValue(
                    key=key,
                    value=actual_value,
                    source=source,
                    encrypted=encrypt
                )
            
            # Notify change listeners
            self._notify_change_listeners(key, old_value, value)
            
            logger.info(f"Configuration updated: {key} (source: {source.value})")
            return True
    
    def is_feature_enabled(self, feature_name: str, user_id: str = "anonymous", 
                          user_groups: Optional[List[str]] = None) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            feature_name: Feature flag name.
            user_id: User identifier.
            user_groups: User groups.
            
        Returns:
            True if feature is enabled.
        """
        with self.lock:
            if feature_name not in self.feature_flags:
                logger.warning(f"Feature flag not found: {feature_name}")
                return False
            
            feature_flag = self.feature_flags[feature_name]
            return feature_flag.is_enabled_for_user(user_id, user_groups)
    
    def get_all_configuration(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Args:
            include_secrets: Whether to include decrypted secrets.
            
        Returns:
            Dictionary of all configuration values.
        """
        with self.lock:
            result = {}
            
            for key, config_value in self.configuration.items():
                if config_value.encrypted and not include_secrets:
                    result[key] = "***ENCRYPTED***"
                elif config_value.encrypted and include_secrets:
                    result[key] = self.secret_manager.decrypt_secret(config_value.value)
                else:
                    result[key] = config_value.value
            
            return result
    
    def get_configuration_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration metadata.
        
        Returns:
            Configuration metadata for all values.
        """
        with self.lock:
            return {key: config.to_dict() for key, config in self.configuration.items()}
    
    def validate_all_configuration(self) -> List[str]:
        """
        Validate all configuration against schemas.
        
        Returns:
            List of validation errors.
        """
        errors = []
        
        with self.lock:
            # Check required configurations
            for schema_name, schema in self.schemas.items():
                if schema.required and schema_name not in self.configuration:
                    errors.append(f"Required configuration missing: {schema_name}")
            
            # Validate existing configurations
            for key, config_value in self.configuration.items():
                if key in self.schemas:
                    is_valid, error = self.schemas[key].validate(config_value.value)
                    if not is_valid:
                        errors.append(error)
        
        return errors
    
    def reload_configuration(self):
        """Reload configuration from all providers."""
        with self.lock:
            self.configuration.clear()
            self._load_all_configurations()
        
        logger.info("Configuration reloaded from all providers")
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """
        Add configuration change listener.
        
        Args:
            listener: Function to call when configuration changes.
        """
        self.change_listeners.append(listener)
    
    def _load_all_configurations(self):
        """Load configuration from all providers."""
        for provider in self.providers:
            self._load_provider_configuration(provider)
    
    def _load_provider_configuration(self, provider: ConfigurationProvider):
        """Load configuration from specific provider."""
        try:
            config = provider.load_configuration()
            
            # Determine source type
            source = ConfigurationSource.MEMORY
            if isinstance(provider, FileConfigurationProvider):
                source = ConfigurationSource.FILE
            elif isinstance(provider, EnvironmentConfigurationProvider):
                source = ConfigurationSource.ENVIRONMENT
            
            # Add configurations
            for key, value in config.items():
                if key not in self.configuration:  # Don't override existing values
                    self.configuration[key] = ConfigurationValue(
                        key=key,
                        value=value,
                        source=source
                    )
        
        except Exception as e:
            logger.error(f"Error loading configuration from provider: {e}")
    
    def _setup_watchers(self):
        """Setup configuration watchers."""
        for provider in self.providers:
            try:
                provider.watch_changes(self._on_configuration_change)
            except Exception as e:
                logger.error(f"Error setting up watcher for provider: {e}")
    
    def _on_configuration_change(self, new_config: Dict[str, Any]):
        """Handle configuration changes from providers."""
        with self.lock:
            for key, value in new_config.items():
                old_value = self.configuration[key].value if key in self.configuration else None
                
                if key in self.configuration:
                    self.configuration[key].value = value
                    self.configuration[key].last_updated = datetime.now()
                    self.configuration[key].version += 1
                else:
                    self.configuration[key] = ConfigurationValue(
                        key=key,
                        value=value,
                        source=ConfigurationSource.FILE  # Assuming file changes
                    )
                
                if old_value != value:
                    self._notify_change_listeners(key, old_value, value)
        
        logger.info("Configuration updated from provider changes")
    
    def _notify_change_listeners(self, key: str, old_value: Any, new_value: Any):
        """Notify configuration change listeners."""
        for listener in self.change_listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in configuration change listener: {e}")


def create_configuration_manager(environment: str = None) -> ConfigurationManager:
    """
    Create configuration manager with default setup.
    
    Args:
        environment: Environment name.
        
    Returns:
        Configured configuration manager.
    """
    if environment is None:
        environment = os.getenv("FUSION_ENV", "development")
    
    # Create providers
    providers = [
        EnvironmentConfigurationProvider("FUSION_"),
        FileConfigurationProvider(f"config/{environment}.yaml")
    ]
    
    # Create manager
    manager = ConfigurationManager(environment, providers)
    
    # Add default schemas
    _add_default_schemas(manager)
    
    # Add default feature flags
    _add_default_feature_flags(manager)
    
    return manager


def _add_default_schemas(manager: ConfigurationManager):
    """Add default configuration schemas."""
    schemas = [
        ConfigurationSchema(
            name="database_url",
            type=ConfigurationType.STRING,
            required=True,
            description="Database connection URL"
        ),
        ConfigurationSchema(
            name="log_level",
            type=ConfigurationType.STRING,
            required=False,
            default="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="Logging level"
        ),
        ConfigurationSchema(
            name="api_rate_limit",
            type=ConfigurationType.INTEGER,
            required=False,
            default=1000,
            min_value=1,
            max_value=10000,
            description="API rate limit per hour"
        ),
        ConfigurationSchema(
            name="enable_debug",
            type=ConfigurationType.BOOLEAN,
            required=False,
            default=False,
            description="Enable debug mode"
        ),
        ConfigurationSchema(
            name="fusion_safety_threshold",
            type=ConfigurationType.FLOAT,
            required=False,
            default=1e21,
            min_value=1e18,
            max_value=1e24,
            description="Fusion safety threshold for triple product"
        )
    ]
    
    for schema in schemas:
        manager.add_schema(schema)


def _add_default_feature_flags(manager: ConfigurationManager):
    """Add default feature flags."""
    feature_flags = [
        FeatureFlag(
            name="advanced_ml_models",
            enabled=True,
            description="Enable advanced ML models for fusion prediction",
            rollout_percentage=100.0
        ),
        FeatureFlag(
            name="real_time_monitoring",
            enabled=True,
            description="Enable real-time monitoring and alerts",
            rollout_percentage=50.0
        ),
        FeatureFlag(
            name="experimental_algorithms",
            enabled=False,
            description="Enable experimental fusion algorithms",
            rollout_percentage=10.0,
            user_groups=["researchers", "beta_testers"]
        ),
        FeatureFlag(
            name="enhanced_security",
            enabled=True,
            description="Enable enhanced security features",
            rollout_percentage=100.0
        )
    ]
    
    for flag in feature_flags:
        manager.add_feature_flag(flag)