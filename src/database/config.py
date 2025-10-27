"""
Database configuration management and environment setup.

This module provides utilities for managing database configurations,
environment variables, and connection settings for different deployment environments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatabaseEnvironment:
    """Database environment configuration."""
    
    name: str
    description: str
    postgresql_config: Dict[str, Any]
    mongodb_config: Dict[str, Any]
    redis_config: Dict[str, Any]
    connection_settings: Dict[str, Any]
    
    def validate(self) -> List[str]:
        """
        Validate environment configuration.
        
        Returns:
            List of validation errors.
        """
        errors = []
        
        # Validate PostgreSQL config
        required_pg_fields = ['host', 'port', 'database', 'username', 'password']
        for field in required_pg_fields:
            if field not in self.postgresql_config:
                errors.append(f"Missing PostgreSQL field: {field}")
        
        # Validate MongoDB config
        required_mongo_fields = ['host', 'port', 'database']
        for field in required_mongo_fields:
            if field not in self.mongodb_config:
                errors.append(f"Missing MongoDB field: {field}")
        
        # Validate Redis config
        required_redis_fields = ['host', 'port', 'db']
        for field in required_redis_fields:
            if field not in self.redis_config:
                errors.append(f"Missing Redis field: {field}")
        
        return errors


class ConfigManager:
    """
    Configuration manager for database settings.
    
    Handles loading configurations from multiple sources,
    environment-specific settings, and configuration validation.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
        """
        self.config_dir = Path(config_dir)
        self.environments: Dict[str, DatabaseEnvironment] = {}
        self.current_environment = "development"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info(f"ConfigManager initialized with config_dir: {config_dir}")
    
    def load_environments(self):
        """Load all environment configurations."""
        # Load from config files
        self._load_from_files()
        
        # Override with environment variables
        self._load_from_env_vars()
        
        # Validate configurations
        self._validate_configurations()
        
        logger.info(f"Loaded {len(self.environments)} environment configurations")
    
    def _load_from_files(self):
        """Load configurations from files."""
        # Load main database config
        main_config_path = self.config_dir / "database.json"
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                main_config = json.load(f)
                self._parse_main_config(main_config)
        
        # Load environment-specific configs
        env_configs_dir = self.config_dir / "environments"
        if env_configs_dir.exists():
            for config_file in env_configs_dir.glob("*.json"):
                env_name = config_file.stem
                with open(config_file, 'r') as f:
                    env_config = json.load(f)
                    self._parse_env_config(env_name, env_config)
        
        # Load YAML configs if present
        for config_file in self.config_dir.glob("*.yaml"):
            self._load_yaml_config(config_file)
        
        logger.debug("Loaded configurations from files")
    
    def _parse_main_config(self, config: Dict[str, Any]):
        """Parse main configuration file."""
        environments = config.get('environments', {})
        
        for env_name, env_config in environments.items():
            self.environments[env_name] = DatabaseEnvironment(
                name=env_name,
                description=env_config.get('description', ''),
                postgresql_config=env_config.get('postgresql', {}),
                mongodb_config=env_config.get('mongodb', {}),
                redis_config=env_config.get('redis', {}),
                connection_settings=env_config.get('connection_settings', {})
            )
    
    def _parse_env_config(self, env_name: str, config: Dict[str, Any]):
        """Parse environment-specific configuration."""
        self.environments[env_name] = DatabaseEnvironment(
            name=env_name,
            description=config.get('description', ''),
            postgresql_config=config.get('postgresql', {}),
            mongodb_config=config.get('mongodb', {}),
            redis_config=config.get('redis', {}),
            connection_settings=config.get('connection_settings', {})
        )
    
    def _load_yaml_config(self, config_file: Path):
        """Load YAML configuration file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'environments' in config:
                for env_name, env_config in config['environments'].items():
                    self._parse_env_config(env_name, env_config)
                    
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML configs")
        except Exception as e:
            logger.error(f"Failed to load YAML config {config_file}: {e}")
    
    def _load_from_env_vars(self):
        """Load and override configurations from environment variables."""
        env_name = os.getenv('DATABASE_ENVIRONMENT', self.current_environment)
        
        # Create or update environment from env vars
        if env_name not in self.environments:
            self.environments[env_name] = DatabaseEnvironment(
                name=env_name,
                description=f"Environment from env vars: {env_name}",
                postgresql_config={},
                mongodb_config={},
                redis_config={},
                connection_settings={}
            )
        
        env_config = self.environments[env_name]
        
        # PostgreSQL overrides
        pg_overrides = {
            'host': os.getenv('POSTGRESQL_HOST'),
            'port': int(os.getenv('POSTGRESQL_PORT', '5432')),
            'database': os.getenv('POSTGRESQL_DATABASE'),
            'username': os.getenv('POSTGRESQL_USERNAME'),
            'password': os.getenv('POSTGRESQL_PASSWORD'),
            'pool_size': int(os.getenv('POSTGRESQL_POOL_SIZE', '20')),
            'max_overflow': int(os.getenv('POSTGRESQL_MAX_OVERFLOW', '40'))
        }
        
        # Apply non-None values
        for key, value in pg_overrides.items():
            if value is not None:
                env_config.postgresql_config[key] = value
        
        # MongoDB overrides
        mongo_overrides = {
            'host': os.getenv('MONGODB_HOST'),
            'port': int(os.getenv('MONGODB_PORT', '27017')),
            'database': os.getenv('MONGODB_DATABASE'),
            'username': os.getenv('MONGODB_USERNAME'),
            'password': os.getenv('MONGODB_PASSWORD'),
            'auth_source': os.getenv('MONGODB_AUTH_SOURCE', 'admin')
        }
        
        for key, value in mongo_overrides.items():
            if value is not None:
                env_config.mongodb_config[key] = value
        
        # Redis overrides
        redis_overrides = {
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD')
        }
        
        for key, value in redis_overrides.items():
            if value is not None:
                env_config.redis_config[key] = value
        
        # Connection settings overrides
        connection_overrides = {
            'timeout': int(os.getenv('CONNECTION_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3')),
            'enable_ssl': os.getenv('ENABLE_SSL', 'false').lower() == 'true'
        }
        
        for key, value in connection_overrides.items():
            if value is not None:
                env_config.connection_settings[key] = value
        
        self.current_environment = env_name
        logger.debug(f"Applied environment variable overrides for: {env_name}")
    
    def _validate_configurations(self):
        """Validate all loaded configurations."""
        for env_name, env_config in self.environments.items():
            errors = env_config.validate()
            if errors:
                logger.warning(f"Configuration errors in {env_name}: {errors}")
            else:
                logger.debug(f"Configuration valid for {env_name}")
    
    def get_environment(self, env_name: Optional[str] = None) -> DatabaseEnvironment:
        """
        Get environment configuration.
        
        Args:
            env_name: Environment name. Uses current if None.
            
        Returns:
            Environment configuration.
        """
        if env_name is None:
            env_name = self.current_environment
        
        if env_name not in self.environments:
            raise ValueError(f"Environment not found: {env_name}")
        
        return self.environments[env_name]
    
    def set_current_environment(self, env_name: str):
        """
        Set current environment.
        
        Args:
            env_name: Environment name.
        """
        if env_name not in self.environments:
            raise ValueError(f"Environment not found: {env_name}")
        
        self.current_environment = env_name
        logger.info(f"Set current environment to: {env_name}")
    
    def create_default_configs(self):
        """Create default configuration files."""
        # Create main database config
        main_config = {
            "environments": {
                "development": {
                    "description": "Development environment",
                    "postgresql": {
                        "host": "localhost",
                        "port": 5432,
                        "database": "fusion_analyzer_dev",
                        "username": "fusion_dev",
                        "password": "dev_password",
                        "pool_size": 10,
                        "max_overflow": 20
                    },
                    "mongodb": {
                        "host": "localhost",
                        "port": 27017,
                        "database": "fusion_analyzer_dev"
                    },
                    "redis": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0
                    },
                    "connection_settings": {
                        "timeout": 30,
                        "retry_attempts": 3,
                        "enable_ssl": False
                    }
                },
                "testing": {
                    "description": "Testing environment",
                    "postgresql": {
                        "host": "localhost",
                        "port": 5432,
                        "database": "fusion_analyzer_test",
                        "username": "fusion_test",
                        "password": "test_password",
                        "pool_size": 5,
                        "max_overflow": 10
                    },
                    "mongodb": {
                        "host": "localhost",
                        "port": 27017,
                        "database": "fusion_analyzer_test"
                    },
                    "redis": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 1
                    },
                    "connection_settings": {
                        "timeout": 15,
                        "retry_attempts": 1,
                        "enable_ssl": False
                    }
                },
                "production": {
                    "description": "Production environment",
                    "postgresql": {
                        "host": "${POSTGRESQL_HOST}",
                        "port": 5432,
                        "database": "fusion_analyzer_prod",
                        "username": "${POSTGRESQL_USERNAME}",
                        "password": "${POSTGRESQL_PASSWORD}",
                        "pool_size": 30,
                        "max_overflow": 60
                    },
                    "mongodb": {
                        "host": "${MONGODB_HOST}",
                        "port": 27017,
                        "database": "fusion_analyzer_prod",
                        "username": "${MONGODB_USERNAME}",
                        "password": "${MONGODB_PASSWORD}"
                    },
                    "redis": {
                        "host": "${REDIS_HOST}",
                        "port": 6379,
                        "db": 0,
                        "password": "${REDIS_PASSWORD}"
                    },
                    "connection_settings": {
                        "timeout": 60,
                        "retry_attempts": 5,
                        "enable_ssl": True
                    }
                }
            }
        }
        
        # Write main config
        main_config_path = self.config_dir / "database.json"
        with open(main_config_path, 'w') as f:
            json.dump(main_config, f, indent=2)
        
        # Create environment-specific directory
        env_dir = self.config_dir / "environments"
        env_dir.mkdir(exist_ok=True)
        
        # Create .env example
        env_example_path = self.config_dir / ".env.example"
        env_example_content = """# Database Environment Configuration
DATABASE_ENVIRONMENT=development

# PostgreSQL Configuration
POSTGRESQL_HOST=localhost
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=fusion_analyzer
POSTGRESQL_USERNAME=fusion_user
POSTGRESQL_PASSWORD=fusion_password
POSTGRESQL_POOL_SIZE=20
POSTGRESQL_MAX_OVERFLOW=40

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=fusion_analyzer
MONGODB_USERNAME=
MONGODB_PASSWORD=
MONGODB_AUTH_SOURCE=admin

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Connection Settings
CONNECTION_TIMEOUT=30
RETRY_ATTEMPTS=3
ENABLE_SSL=false
"""
        
        with open(env_example_path, 'w') as f:
            f.write(env_example_content)
        
        logger.info("Created default configuration files")
    
    def export_config(self, env_name: str, output_path: str):
        """
        Export environment configuration to file.
        
        Args:
            env_name: Environment name.
            output_path: Output file path.
        """
        if env_name not in self.environments:
            raise ValueError(f"Environment not found: {env_name}")
        
        env_config = self.environments[env_name]
        
        output_data = {
            'environment': asdict(env_config),
            'exported_at': str(datetime.now()),
            'exported_by': 'fusion_analyzer_config_manager'
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Exported {env_name} configuration to {output_path}")
    
    def list_environments(self) -> List[str]:
        """
        List available environments.
        
        Returns:
            List of environment names.
        """
        return list(self.environments.keys())
    
    def get_connection_string(self, 
                            env_name: Optional[str] = None,
                            database_type: str = 'postgresql') -> str:
        """
        Get database connection string.
        
        Args:
            env_name: Environment name.
            database_type: Database type (postgresql, mongodb, redis).
            
        Returns:
            Connection string.
        """
        env_config = self.get_environment(env_name)
        
        if database_type == 'postgresql':
            pg_config = env_config.postgresql_config
            return (
                f"postgresql://{pg_config['username']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            )
        
        elif database_type == 'mongodb':
            mongo_config = env_config.mongodb_config
            if mongo_config.get('username') and mongo_config.get('password'):
                return (
                    f"mongodb://{mongo_config['username']}:{mongo_config['password']}"
                    f"@{mongo_config['host']}:{mongo_config['port']}/{mongo_config['database']}"
                    f"?authSource={mongo_config.get('auth_source', 'admin')}"
                )
            else:
                return f"mongodb://{mongo_config['host']}:{mongo_config['port']}"
        
        elif database_type == 'redis':
            redis_config = env_config.redis_config
            if redis_config.get('password'):
                return (
                    f"redis://:{redis_config['password']}"
                    f"@{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
                )
            else:
                return f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
        
        else:
            raise ValueError(f"Unsupported database type: {database_type}")


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_environments()
    return _config_manager


def initialize_config_manager(config_dir: str = "config") -> ConfigManager:
    """
    Initialize the global configuration manager.
    
    Args:
        config_dir: Configuration directory.
        
    Returns:
        Configuration manager instance.
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    _config_manager.load_environments()
    return _config_manager