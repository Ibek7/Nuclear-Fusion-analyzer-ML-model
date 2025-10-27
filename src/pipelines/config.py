"""
Configuration Management for Data Pipelines.

This module provides:
- Pipeline configuration validation
- Environment-specific configurations
- Configuration templates
- Dynamic configuration loading
- Configuration versioning
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    name: str
    type: str  # file, database, api, stream
    enabled: bool = True
    connection: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    schedule: Optional[str] = None
    retry_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.connection is None:
            self.connection = {}
        if self.retry_config is None:
            self.retry_config = {"max_retries": 3, "retry_delay": 5}


@dataclass
class TransformationConfig:
    """Configuration for data transformations."""
    name: str
    type: str  # normalize, calculate, filter, aggregate
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    enabled: bool = True
    rules: Optional[List[Dict[str, Any]]] = None
    on_failure: str = "stop"  # stop, warn, skip
    custom_validators: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.rules is None:
            self.rules = []
        if self.custom_validators is None:
            self.custom_validators = []


@dataclass
class OutputConfig:
    """Configuration for data outputs."""
    name: str
    type: str  # file, database, api, stream
    enabled: bool = True
    connection: Optional[Dict[str, Any]] = None
    format: str = "csv"
    compression: Optional[str] = None
    partitioning: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.connection is None:
            self.connection = {}


@dataclass
class MonitoringConfig:
    """Configuration for pipeline monitoring."""
    enabled: bool = True
    metrics: Optional[List[str]] = None
    alerts: Optional[Dict[str, Any]] = None
    logging_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.metrics is None:
            self.metrics = ["throughput", "latency", "errors", "data_quality"]
        if self.alerts is None:
            self.alerts = {}


@dataclass
class PerformanceConfig:
    """Configuration for pipeline performance."""
    batch_size: int = 1000
    max_workers: int = 4
    memory_limit: Optional[str] = None
    timeout: int = 300
    parallel_processing: bool = True
    caching: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.caching is None:
            self.caching = {"enabled": True, "ttl": 3600}


@dataclass
class StreamingConfig:
    """Configuration for streaming pipelines."""
    enabled: bool = False
    kafka: Optional[Dict[str, Any]] = None
    websocket: Optional[Dict[str, Any]] = None
    buffer_size: int = 10000
    processing_interval: int = 1000  # milliseconds
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.kafka is None:
            self.kafka = {"enabled": False}
        if self.websocket is None:
            self.websocket = {"enabled": False}


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str
    version: str
    description: Optional[str] = None
    environment: str = "development"
    
    # Core components
    sources: Optional[List[DataSourceConfig]] = None
    transformations: Optional[List[TransformationConfig]] = None
    validation: Optional[ValidationConfig] = None
    outputs: Optional[List[OutputConfig]] = None
    
    # Configuration sections
    monitoring: Optional[MonitoringConfig] = None
    performance: Optional[PerformanceConfig] = None
    streaming: Optional[StreamingConfig] = None
    
    # Scheduling
    schedule: Optional[str] = None
    dependencies: Optional[List[str]] = None
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sources is None:
            self.sources = []
        if self.transformations is None:
            self.transformations = []
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.outputs is None:
            self.outputs = []
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.streaming is None:
            self.streaming = StreamingConfig()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


class ConfigurationManager:
    """Manages pipeline configurations."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Environment-specific configuration
        self.environment = os.getenv("FUSION_ENV", "development")
        
        logger.info(f"ConfigurationManager initialized for environment: {self.environment}")
    
    def save_config(self, config: PipelineConfig, filename: Optional[str] = None) -> str:
        """
        Save pipeline configuration to file.
        
        Args:
            config: Pipeline configuration.
            filename: Output filename (optional).
            
        Returns:
            Path to saved configuration file.
        """
        if filename is None:
            filename = f"{config.name}_{config.environment}.yaml"
        
        file_path = self.config_dir / filename
        
        # Update metadata
        config.updated_at = datetime.now().isoformat()
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save as YAML
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved: {file_path}")
        return str(file_path)
    
    def load_config(self, filename: str) -> PipelineConfig:
        """
        Load pipeline configuration from file.
        
        Args:
            filename: Configuration filename.
            
        Returns:
            Loaded pipeline configuration.
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load YAML
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclasses
        config = self._dict_to_config(config_dict)
        
        logger.info(f"Configuration loaded: {file_path}")
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        # Convert sources
        sources = []
        if config_dict.get("sources"):
            for source_dict in config_dict["sources"]:
                sources.append(DataSourceConfig(**source_dict))
        
        # Convert transformations
        transformations = []
        if config_dict.get("transformations"):
            for trans_dict in config_dict["transformations"]:
                transformations.append(TransformationConfig(**trans_dict))
        
        # Convert outputs
        outputs = []
        if config_dict.get("outputs"):
            for output_dict in config_dict["outputs"]:
                outputs.append(OutputConfig(**output_dict))
        
        # Convert nested configurations
        validation = None
        if config_dict.get("validation"):
            validation = ValidationConfig(**config_dict["validation"])
        
        monitoring = None
        if config_dict.get("monitoring"):
            monitoring = MonitoringConfig(**config_dict["monitoring"])
        
        performance = None
        if config_dict.get("performance"):
            performance = PerformanceConfig(**config_dict["performance"])
        
        streaming = None
        if config_dict.get("streaming"):
            streaming = StreamingConfig(**config_dict["streaming"])
        
        # Create main configuration
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["sources", "transformations", "outputs", "validation", 
                                   "monitoring", "performance", "streaming"]}
        
        return PipelineConfig(
            sources=sources,
            transformations=transformations,
            outputs=outputs,
            validation=validation,
            monitoring=monitoring,
            performance=performance,
            streaming=streaming,
            **main_config
        )
    
    def list_configs(self) -> List[str]:
        """
        List available configuration files.
        
        Returns:
            List of configuration filenames.
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.name)
        
        return sorted(config_files)
    
    def validate_config(self, config: PipelineConfig) -> List[str]:
        """
        Validate pipeline configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        # Required fields
        if not config.name:
            errors.append("Pipeline name is required")
        
        if not config.version:
            errors.append("Pipeline version is required")
        
        # At least one source required
        if not config.sources:
            errors.append("At least one data source is required")
        
        # At least one output required
        if not config.outputs:
            errors.append("At least one output is required")
        
        # Validate sources
        for i, source in enumerate(config.sources):
            if not source.name:
                errors.append(f"Source {i}: name is required")
            
            if source.type not in ["file", "database", "api", "stream"]:
                errors.append(f"Source {i}: invalid type '{source.type}'")
            
            if source.type == "database" and not source.connection:
                errors.append(f"Source {i}: database connection configuration required")
        
        # Validate transformations
        for i, transformation in enumerate(config.transformations):
            if not transformation.name:
                errors.append(f"Transformation {i}: name is required")
            
            if transformation.type not in ["normalize", "calculate", "filter", "aggregate", "custom"]:
                errors.append(f"Transformation {i}: invalid type '{transformation.type}'")
        
        # Validate outputs
        for i, output in enumerate(config.outputs):
            if not output.name:
                errors.append(f"Output {i}: name is required")
            
            if output.type not in ["file", "database", "api", "stream"]:
                errors.append(f"Output {i}: invalid type '{output.type}'")
        
        # Validate performance settings
        if config.performance:
            if config.performance.batch_size <= 0:
                errors.append("Performance: batch_size must be positive")
            
            if config.performance.max_workers <= 0:
                errors.append("Performance: max_workers must be positive")
            
            if config.performance.timeout <= 0:
                errors.append("Performance: timeout must be positive")
        
        return errors
    
    def create_template(self, pipeline_name: str, pipeline_type: str = "batch") -> PipelineConfig:
        """
        Create configuration template.
        
        Args:
            pipeline_name: Name of the pipeline.
            pipeline_type: Type of pipeline (batch, streaming).
            
        Returns:
            Template configuration.
        """
        if pipeline_type == "streaming":
            return self._create_streaming_template(pipeline_name)
        else:
            return self._create_batch_template(pipeline_name)
    
    def _create_batch_template(self, pipeline_name: str) -> PipelineConfig:
        """Create batch pipeline template."""
        return PipelineConfig(
            name=pipeline_name,
            version="1.0.0",
            description=f"Batch processing pipeline for {pipeline_name}",
            environment=self.environment,
            sources=[
                DataSourceConfig(
                    name="fusion_data_source",
                    type="file",
                    connection={
                        "path": "data/input/fusion_data.csv"
                    }
                )
            ],
            transformations=[
                TransformationConfig(
                    name="normalize_plasma_parameters",
                    type="normalize",
                    parameters={
                        "columns": ["plasma_temperature", "plasma_density"]
                    }
                ),
                TransformationConfig(
                    name="calculate_fusion_metrics",
                    type="calculate",
                    parameters={
                        "metrics": ["triple_product", "beta", "fusion_power"]
                    }
                )
            ],
            outputs=[
                OutputConfig(
                    name="processed_data",
                    type="file",
                    connection={
                        "path": "data/output/processed_fusion_data.csv"
                    }
                )
            ],
            schedule="@daily",
            tags=["fusion", "batch", "analytics"]
        )
    
    def _create_streaming_template(self, pipeline_name: str) -> PipelineConfig:
        """Create streaming pipeline template."""
        return PipelineConfig(
            name=pipeline_name,
            version="1.0.0",
            description=f"Real-time streaming pipeline for {pipeline_name}",
            environment=self.environment,
            sources=[
                DataSourceConfig(
                    name="realtime_sensors",
                    type="stream",
                    connection={
                        "kafka_topic": "fusion_sensor_data",
                        "bootstrap_servers": ["localhost:9092"]
                    }
                )
            ],
            transformations=[
                TransformationConfig(
                    name="realtime_validation",
                    type="validate",
                    parameters={
                        "physics_constraints": True
                    }
                ),
                TransformationConfig(
                    name="anomaly_detection",
                    type="detect",
                    parameters={
                        "algorithm": "isolation_forest",
                        "threshold": 0.1
                    }
                )
            ],
            outputs=[
                OutputConfig(
                    name="alerts_stream",
                    type="stream",
                    connection={
                        "kafka_topic": "fusion_alerts",
                        "bootstrap_servers": ["localhost:9092"]
                    }
                ),
                OutputConfig(
                    name="processed_stream",
                    type="database",
                    connection={
                        "host": "localhost",
                        "database": "fusion_db",
                        "table": "realtime_data"
                    }
                )
            ],
            streaming=StreamingConfig(
                enabled=True,
                kafka={
                    "enabled": True,
                    "bootstrap_servers": ["localhost:9092"],
                    "topic": "fusion_data"
                },
                websocket={
                    "enabled": True,
                    "host": "localhost",
                    "port": 8765
                }
            ),
            tags=["fusion", "streaming", "realtime", "alerts"]
        )
    
    def get_environment_config(self, base_config: PipelineConfig, environment: str) -> PipelineConfig:
        """
        Get environment-specific configuration.
        
        Args:
            base_config: Base configuration.
            environment: Target environment.
            
        Returns:
            Environment-specific configuration.
        """
        env_config = PipelineConfig(**asdict(base_config))
        env_config.environment = environment
        
        # Environment-specific modifications
        if environment == "production":
            # Production optimizations
            if env_config.performance:
                env_config.performance.batch_size = min(env_config.performance.batch_size * 2, 5000)
                env_config.performance.max_workers = min(env_config.performance.max_workers * 2, 16)
            
            if env_config.monitoring:
                env_config.monitoring.logging_level = "INFO"
                env_config.monitoring.alerts = {
                    "enabled": True,
                    "thresholds": {
                        "error_rate": 0.01,
                        "latency": 5000,
                        "throughput": 100
                    }
                }
        
        elif environment == "development":
            # Development optimizations
            if env_config.performance:
                env_config.performance.batch_size = max(env_config.performance.batch_size // 2, 100)
                env_config.performance.max_workers = max(env_config.performance.max_workers // 2, 1)
            
            if env_config.monitoring:
                env_config.monitoring.logging_level = "DEBUG"
        
        elif environment == "testing":
            # Testing optimizations
            if env_config.performance:
                env_config.performance.batch_size = 10
                env_config.performance.max_workers = 1
                env_config.performance.timeout = 60
        
        return env_config


def create_configuration_manager(config_dir: str = "config") -> ConfigurationManager:
    """
    Create configuration manager.
    
    Args:
        config_dir: Configuration directory.
        
    Returns:
        Configuration manager instance.
    """
    return ConfigurationManager(config_dir)