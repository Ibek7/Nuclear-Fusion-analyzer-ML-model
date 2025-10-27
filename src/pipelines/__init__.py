"""
Advanced Data Pipeline and ETL System for Nuclear Fusion Analysis.

This module provides:
- Data ingestion from multiple sources
- Data transformation and enrichment
- Data validation and quality checks
- Pipeline orchestration and scheduling
- Apache Airflow integration
- Stream processing capabilities
- Error handling and recovery
- Data lineage tracking
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import tempfile
import shutil

# Data processing
import pandas as pd
import numpy as np

# Async file operations
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

# Apache Airflow (optional)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.utils.dates import days_ago
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

# Stream processing
try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

# Database connectors
try:
    import psycopg2
    import pymongo
    HAS_DB_CONNECTORS = True
except ImportError:
    HAS_DB_CONNECTORS = False

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    XML = "xml"
    BINARY = "binary"


class SourceType(Enum):
    """Data source types."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    S3 = "s3"
    FTP = "ftp"


@dataclass
class DataSource:
    """Data source configuration."""
    
    name: str
    source_type: SourceType
    connection_config: Dict[str, Any]
    data_format: DataFormat = DataFormat.JSON
    schema: Optional[Dict[str, Any]] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "connection_config": self.connection_config,
            "data_format": self.data_format.value,
            "schema": self.schema,
            "enabled": self.enabled
        }


@dataclass
class TransformationStep:
    """Data transformation step."""
    
    name: str
    transform_function: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to data."""
        try:
            # Validate required columns
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Apply transformation
            result = self.transform_function(data, **self.config)
            
            # Validate output
            if self.output_columns:
                missing_output = set(self.output_columns) - set(result.columns)
                if missing_output:
                    logger.warning(f"Expected output columns missing: {missing_output}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transformation '{self.name}' failed: {e}")
            raise


@dataclass
class PipelineRun:
    """Pipeline execution run information."""
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.PENDING
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate run duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.records_processed + self.records_failed
        if total == 0:
            return 1.0
        return self.records_processed / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "success_rate": self.success_rate,
            "duration": str(self.duration) if self.duration else None,
            "error_message": self.error_message,
            "metrics": self.metrics
        }


class DataConnector(ABC):
    """Abstract data connector interface."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source."""
        pass
    
    @abstractmethod
    async def read_data(self, **kwargs) -> pd.DataFrame:
        """Read data from source."""
        pass
    
    @abstractmethod
    async def write_data(self, data: pd.DataFrame, **kwargs) -> bool:
        """Write data to destination."""
        pass


class FileConnector(DataConnector):
    """File-based data connector."""
    
    def __init__(self, base_path: str):
        """
        Initialize file connector.
        
        Args:
            base_path: Base directory path for files.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileConnector initialized with path: {self.base_path}")
    
    async def connect(self) -> bool:
        """Check if base path exists and is accessible."""
        return self.base_path.exists() and self.base_path.is_dir()
    
    async def disconnect(self):
        """No-op for file connector."""
        pass
    
    async def read_data(
        self,
        filename: str,
        data_format: DataFormat = DataFormat.CSV,
        **kwargs
    ) -> pd.DataFrame:
        """Read data from file."""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if data_format == DataFormat.CSV:
                return pd.read_csv(file_path, **kwargs)
            elif data_format == DataFormat.JSON:
                return pd.read_json(file_path, **kwargs)
            elif data_format == DataFormat.PARQUET:
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {data_format}")
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    async def write_data(
        self,
        data: pd.DataFrame,
        filename: str,
        data_format: DataFormat = DataFormat.CSV,
        **kwargs
    ) -> bool:
        """Write data to file."""
        file_path = self.base_path / filename
        
        try:
            if data_format == DataFormat.CSV:
                data.to_csv(file_path, index=False, **kwargs)
            elif data_format == DataFormat.JSON:
                data.to_json(file_path, **kwargs)
            elif data_format == DataFormat.PARQUET:
                data.to_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {data_format}")
            
            logger.info(f"Data written to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False


class DatabaseConnector(DataConnector):
    """Database data connector."""
    
    def __init__(self, connection_string: str, db_type: str = "postgresql"):
        """
        Initialize database connector.
        
        Args:
            connection_string: Database connection string.
            db_type: Database type (postgresql, mongodb).
        """
        self.connection_string = connection_string
        self.db_type = db_type
        self.connection = None
        
        logger.info(f"DatabaseConnector initialized for {db_type}")
    
    async def connect(self) -> bool:
        """Connect to database."""
        try:
            if self.db_type == "postgresql":
                import psycopg2
                self.connection = psycopg2.connect(self.connection_string)
                return True
            elif self.db_type == "mongodb":
                import pymongo
                self.connection = pymongo.MongoClient(self.connection_string)
                return True
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def read_data(self, query: str, **kwargs) -> pd.DataFrame:
        """Read data using SQL query."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        try:
            if self.db_type == "postgresql":
                return pd.read_sql(query, self.connection, **kwargs)
            elif self.db_type == "mongodb":
                # MongoDB queries would need different handling
                raise NotImplementedError("MongoDB read not implemented")
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Database read failed: {e}")
            raise
    
    async def write_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        **kwargs
    ) -> bool:
        """Write data to database table."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        try:
            if self.db_type == "postgresql":
                data.to_sql(table_name, self.connection, if_exists=if_exists, index=False, **kwargs)
                return True
            elif self.db_type == "mongodb":
                # MongoDB writes would need different handling
                raise NotImplementedError("MongoDB write not implemented")
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Database write failed: {e}")
            return False


class APIConnector(DataConnector):
    """REST API data connector."""
    
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize API connector.
        
        Args:
            base_url: Base API URL.
            headers: Default headers for requests.
        """
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.session = None
        
        logger.info(f"APIConnector initialized for {base_url}")
    
    async def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            import httpx
            self.session = httpx.AsyncClient(headers=self.headers, timeout=30.0)
            return True
        except Exception as e:
            logger.error(f"API connector initialization failed: {e}")
            return False
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def read_data(self, endpoint: str, **kwargs) -> pd.DataFrame:
        """Read data from API endpoint."""
        if not self.session:
            raise RuntimeError("Not connected to API")
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = await self.session.get(url, params=kwargs.get('params'))
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
                
        except Exception as e:
            logger.error(f"API read failed: {e}")
            raise
    
    async def write_data(self, data: pd.DataFrame, endpoint: str, **kwargs) -> bool:
        """Write data to API endpoint."""
        if not self.session:
            raise RuntimeError("Not connected to API")
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            # Convert DataFrame to JSON
            data_json = data.to_dict('records')
            
            response = await self.session.post(url, json=data_json, **kwargs)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"API write failed: {e}")
            return False


class DataValidator:
    """Data quality validation and checks."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_rules = self._get_default_rules()
        
        logger.info("DataValidator initialized")
    
    def _get_default_rules(self) -> List[Dict[str, Any]]:
        """Get default validation rules for fusion data."""
        return [
            {
                "name": "temperature_range",
                "type": "range",
                "column": "plasma_temperature",
                "min_value": 0,
                "max_value": 200,  # Million Kelvin
                "severity": "error"
            },
            {
                "name": "density_positive",
                "type": "positive",
                "column": "plasma_density",
                "severity": "error"
            },
            {
                "name": "pressure_range",
                "type": "range",
                "column": "magnetic_pressure",
                "min_value": 0,
                "max_value": 10,  # Tesla
                "severity": "warning"
            },
            {
                "name": "timestamp_recent",
                "type": "temporal",
                "column": "timestamp",
                "max_age_hours": 24,
                "severity": "warning"
            },
            {
                "name": "required_columns",
                "type": "required",
                "columns": ["timestamp", "plasma_temperature", "plasma_density"],
                "severity": "error"
            }
        ]
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against rules.
        
        Args:
            data: Data to validate.
            
        Returns:
            Validation results.
        """
        results = {
            "total_records": len(data),
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "errors": [],
            "warnings": [],
            "rule_results": {}
        }
        
        for rule in self.validation_rules:
            rule_result = self._apply_rule(data, rule)
            results["rule_results"][rule["name"]] = rule_result
            
            if not rule_result["passed"]:
                if rule["severity"] == "error":
                    results["errors"].extend(rule_result["issues"])
                    results["passed"] = False
                elif rule["severity"] == "warning":
                    results["warnings"].extend(rule_result["issues"])
        
        return results
    
    def _apply_rule(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply single validation rule."""
        result = {
            "rule_name": rule["name"],
            "passed": True,
            "issues": [],
            "affected_records": 0
        }
        
        try:
            if rule["type"] == "range":
                column = rule["column"]
                if column in data.columns:
                    invalid_mask = (
                        (data[column] < rule["min_value"]) |
                        (data[column] > rule["max_value"])
                    )
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        result["passed"] = False
                        result["affected_records"] = invalid_count
                        result["issues"].append(
                            f"Column '{column}' has {invalid_count} values outside range "
                            f"[{rule['min_value']}, {rule['max_value']}]"
                        )
            
            elif rule["type"] == "positive":
                column = rule["column"]
                if column in data.columns:
                    negative_mask = data[column] <= 0
                    negative_count = negative_mask.sum()
                    
                    if negative_count > 0:
                        result["passed"] = False
                        result["affected_records"] = negative_count
                        result["issues"].append(
                            f"Column '{column}' has {negative_count} non-positive values"
                        )
            
            elif rule["type"] == "required":
                missing_columns = set(rule["columns"]) - set(data.columns)
                if missing_columns:
                    result["passed"] = False
                    result["issues"].append(f"Missing required columns: {missing_columns}")
            
            elif rule["type"] == "temporal":
                column = rule["column"]
                if column in data.columns:
                    max_age = timedelta(hours=rule["max_age_hours"])
                    cutoff_time = datetime.now(timezone.utc) - max_age
                    
                    # Convert to datetime if needed
                    if data[column].dtype == 'object':
                        data[column] = pd.to_datetime(data[column])
                    
                    old_mask = data[column] < cutoff_time
                    old_count = old_mask.sum()
                    
                    if old_count > 0:
                        result["passed"] = False
                        result["affected_records"] = old_count
                        result["issues"].append(
                            f"Column '{column}' has {old_count} records older than {rule['max_age_hours']} hours"
                        )
        
        except Exception as e:
            result["passed"] = False
            result["issues"].append(f"Validation error: {e}")
        
        return result


class Pipeline:
    """
    Data processing pipeline with transformations and validation.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name.
            description: Pipeline description.
        """
        self.name = name
        self.description = description
        self.sources: List[DataSource] = []
        self.transformations: List[TransformationStep] = []
        self.destinations: List[Tuple[DataConnector, Dict[str, Any]]] = []
        self.validator = DataValidator()
        self.runs: List[PipelineRun] = []
        
        logger.info(f"Pipeline '{name}' initialized")
    
    def add_source(self, source: DataSource):
        """Add data source to pipeline."""
        self.sources.append(source)
        logger.info(f"Source '{source.name}' added to pipeline '{self.name}'")
    
    def add_transformation(self, transformation: TransformationStep):
        """Add transformation step to pipeline."""
        self.transformations.append(transformation)
        logger.info(f"Transformation '{transformation.name}' added to pipeline '{self.name}'")
    
    def add_destination(self, connector: DataConnector, config: Dict[str, Any]):
        """Add destination to pipeline."""
        self.destinations.append((connector, config))
        logger.info(f"Destination added to pipeline '{self.name}'")
    
    async def execute(self, run_config: Optional[Dict[str, Any]] = None) -> PipelineRun:
        """
        Execute pipeline.
        
        Args:
            run_config: Runtime configuration.
            
        Returns:
            Pipeline run information.
        """
        run = PipelineRun(pipeline_name=self.name)
        run.status = PipelineStatus.RUNNING
        self.runs.append(run)
        
        try:
            logger.info(f"Starting pipeline execution: {self.name}")
            
            # Collect data from all sources
            all_data = []
            
            for source in self.sources:
                if not source.enabled:
                    continue
                
                try:
                    # Create connector based on source type
                    connector = self._create_connector(source)
                    
                    # Connect and read data
                    await connector.connect()
                    data = await connector.read_data(**source.connection_config)
                    await connector.disconnect()
                    
                    all_data.append(data)
                    logger.info(f"Read {len(data)} records from source '{source.name}'")
                    
                except Exception as e:
                    logger.error(f"Error reading from source '{source.name}': {e}")
                    run.records_failed += 1
            
            if not all_data:
                raise RuntimeError("No data collected from sources")
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            current_data = combined_data.copy()
            
            logger.info(f"Combined data: {len(current_data)} records")
            
            # Apply transformations
            for transformation in self.transformations:
                try:
                    logger.info(f"Applying transformation: {transformation.name}")
                    current_data = transformation.apply(current_data)
                    logger.info(f"Transformation result: {len(current_data)} records")
                    
                except Exception as e:
                    logger.error(f"Transformation '{transformation.name}' failed: {e}")
                    run.error_message = str(e)
                    run.status = PipelineStatus.FAILED
                    return run
            
            # Validate data
            validation_results = self.validator.validate_data(current_data)
            run.metrics["validation"] = validation_results
            
            if not validation_results["passed"]:
                logger.warning(f"Data validation failed: {validation_results['errors']}")
                if run_config and not run_config.get("continue_on_validation_error", False):
                    run.status = PipelineStatus.FAILED
                    run.error_message = f"Data validation failed: {validation_results['errors']}"
                    return run
            
            # Write to destinations
            for connector, config in self.destinations:
                try:
                    await connector.connect()
                    success = await connector.write_data(current_data, **config)
                    await connector.disconnect()
                    
                    if not success:
                        raise RuntimeError("Write operation failed")
                    
                    logger.info(f"Data written to destination successfully")
                    
                except Exception as e:
                    logger.error(f"Error writing to destination: {e}")
                    run.records_failed += len(current_data)
            
            # Update run metrics
            run.records_processed = len(current_data)
            run.status = PipelineStatus.SUCCESS
            run.end_time = datetime.now(timezone.utc)
            
            logger.info(f"Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline '{self.name}' failed: {e}")
            run.status = PipelineStatus.FAILED
            run.error_message = str(e)
            run.end_time = datetime.now(timezone.utc)
        
        return run
    
    def _create_connector(self, source: DataSource) -> DataConnector:
        """Create appropriate connector for data source."""
        if source.source_type == SourceType.FILE:
            return FileConnector(source.connection_config.get("path", "/tmp"))
        elif source.source_type == SourceType.DATABASE:
            return DatabaseConnector(
                source.connection_config.get("connection_string"),
                source.connection_config.get("db_type", "postgresql")
            )
        elif source.source_type == SourceType.API:
            return APIConnector(
                source.connection_config.get("base_url"),
                source.connection_config.get("headers")
            )
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
    
    def get_run_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pipeline run history."""
        recent_runs = sorted(self.runs, key=lambda r: r.start_time, reverse=True)[:limit]
        return [run.to_dict() for run in recent_runs]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.runs:
            return {"message": "No runs available"}
        
        total_runs = len(self.runs)
        successful_runs = sum(1 for run in self.runs if run.status == PipelineStatus.SUCCESS)
        failed_runs = sum(1 for run in self.runs if run.status == PipelineStatus.FAILED)
        
        total_records = sum(run.records_processed for run in self.runs)
        total_failures = sum(run.records_failed for run in self.runs)
        
        durations = [run.duration.total_seconds() for run in self.runs if run.duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "total_records_processed": total_records,
            "total_record_failures": total_failures,
            "average_duration_seconds": avg_duration,
            "last_run": self.runs[-1].to_dict() if self.runs else None
        }


# Fusion-specific transformation functions
def normalize_plasma_parameters(data: pd.DataFrame, **config) -> pd.DataFrame:
    """Normalize plasma parameters to standard units."""
    result = data.copy()
    
    # Convert temperature to millions of Kelvin if needed
    if 'plasma_temperature' in result.columns:
        if result['plasma_temperature'].max() > 1000:  # Assume Kelvin
            result['plasma_temperature'] = result['plasma_temperature'] / 1e6
    
    # Normalize density to 10^19 m^-3
    if 'plasma_density' in result.columns:
        if result['plasma_density'].max() > 100:  # Assume raw m^-3
            result['plasma_density'] = result['plasma_density'] / 1e19
    
    return result


def calculate_fusion_metrics(data: pd.DataFrame, **config) -> pd.DataFrame:
    """Calculate derived fusion metrics."""
    result = data.copy()
    
    # Calculate pressure if temperature and density available
    if 'plasma_temperature' in result.columns and 'plasma_density' in result.columns:
        # Simplified pressure calculation (kT * n)
        k_boltzmann = 1.38e-23  # J/K
        result['plasma_pressure'] = result['plasma_temperature'] * 1e6 * k_boltzmann * result['plasma_density'] * 1e19
    
    # Calculate confinement time if available
    if 'energy_content' in result.columns and 'power_loss' in result.columns:
        result['confinement_time'] = result['energy_content'] / result['power_loss']
    
    # Calculate beta (plasma pressure / magnetic pressure)
    if 'plasma_pressure' in result.columns and 'magnetic_pressure' in result.columns:
        result['beta'] = result['plasma_pressure'] / result['magnetic_pressure']
    
    return result


def filter_outliers(data: pd.DataFrame, **config) -> pd.DataFrame:
    """Remove statistical outliers from data."""
    result = data.copy()
    columns = config.get('columns', [])
    method = config.get('method', 'iqr')
    threshold = config.get('threshold', 1.5)
    
    for column in columns:
        if column not in result.columns:
            continue
        
        if method == 'iqr':
            Q1 = result[column].quantile(0.25)
            Q3 = result[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (result[column] >= lower_bound) & (result[column] <= upper_bound)
            result = result[mask]
        
        elif method == 'zscore':
            z_scores = np.abs((result[column] - result[column].mean()) / result[column].std())
            result = result[z_scores <= threshold]
    
    return result


class PipelineManager:
    """
    Pipeline management and orchestration system.
    
    Provides pipeline creation, scheduling, and monitoring.
    """
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, Pipeline] = {}
        self.scheduler_running = False
        
        logger.info("PipelineManager initialized")
    
    def create_pipeline(self, name: str, description: str = "") -> Pipeline:
        """Create new pipeline."""
        pipeline = Pipeline(name, description)
        self.pipelines[name] = pipeline
        
        logger.info(f"Pipeline '{name}' created")
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get pipeline by name."""
        return self.pipelines.get(name)
    
    def list_pipelines(self) -> List[str]:
        """List all pipeline names."""
        return list(self.pipelines.keys())
    
    async def execute_pipeline(self, name: str, run_config: Optional[Dict[str, Any]] = None) -> PipelineRun:
        """Execute pipeline by name."""
        pipeline = self.get_pipeline(name)
        if not pipeline:
            raise ValueError(f"Pipeline '{name}' not found")
        
        return await pipeline.execute(run_config)
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pipelines."""
        stats = {}
        
        for name, pipeline in self.pipelines.items():
            stats[name] = pipeline.get_statistics()
        
        return stats
    
    def create_fusion_analysis_pipeline(self) -> Pipeline:
        """Create default fusion analysis pipeline."""
        pipeline = self.create_pipeline(
            "fusion_analysis",
            "Nuclear fusion data analysis pipeline"
        )
        
        # Add data source
        file_source = DataSource(
            name="fusion_data_files",
            source_type=SourceType.FILE,
            connection_config={"path": "/data/fusion", "filename": "plasma_data.csv"},
            data_format=DataFormat.CSV
        )
        pipeline.add_source(file_source)
        
        # Add transformations
        normalize_step = TransformationStep(
            name="normalize_parameters",
            transform_function=normalize_plasma_parameters,
            required_columns=["plasma_temperature", "plasma_density"]
        )
        pipeline.add_transformation(normalize_step)
        
        metrics_step = TransformationStep(
            name="calculate_metrics",
            transform_function=calculate_fusion_metrics,
            required_columns=["plasma_temperature", "plasma_density"]
        )
        pipeline.add_transformation(metrics_step)
        
        outlier_step = TransformationStep(
            name="filter_outliers",
            transform_function=filter_outliers,
            config={
                "columns": ["plasma_temperature", "plasma_density", "beta"],
                "method": "iqr",
                "threshold": 2.0
            }
        )
        pipeline.add_transformation(outlier_step)
        
        # Add destination
        file_connector = FileConnector("/data/processed")
        pipeline.add_destination(file_connector, {
            "filename": "processed_fusion_data.csv",
            "data_format": DataFormat.CSV
        })
        
        return pipeline


def create_pipeline_manager() -> PipelineManager:
    """
    Create configured pipeline manager.
    
    Returns:
        Configured pipeline manager.
    """
    return PipelineManager()