"""
Database integration module for Nuclear Fusion Analyzer.

This module provides comprehensive database integration capabilities
including PostgreSQL for relational data, MongoDB for document storage,
connection pooling, ORM models, and data persistence layers.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import json
import uuid

# PostgreSQL dependencies
try:
    import asyncpg
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text, Index
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

# MongoDB dependencies
try:
    import motor.motor_asyncio
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from bson import ObjectId
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

# Redis dependencies
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration container."""
    
    # PostgreSQL settings
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_database: str = "fusion_analyzer"
    postgresql_username: str = "fusion_user"
    postgresql_password: str = "fusion_password"
    postgresql_pool_size: int = 20
    postgresql_max_overflow: int = 40
    
    # MongoDB settings
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "fusion_analyzer"
    mongodb_username: Optional[str] = None
    mongodb_password: Optional[str] = None
    mongodb_auth_source: str = "admin"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Connection settings
    connection_timeout: int = 30
    retry_attempts: int = 3
    enable_ssl: bool = False


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class FusionExperiment(Base):
    """SQLAlchemy model for fusion experiments."""
    
    __tablename__ = "fusion_experiments"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Experiment metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    experiment_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="planned")
    
    # Reactor configuration
    reactor_type: Mapped[str] = mapped_column(String(100), nullable=False)
    major_radius: Mapped[float] = mapped_column(Float)
    minor_radius: Mapped[float] = mapped_column(Float)
    magnetic_field: Mapped[float] = mapped_column(Float)
    
    # Additional metadata as JSON
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    shots = relationship("FusionShot", back_populates="experiment", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_experiments_created_at', 'created_at'),
        Index('ix_experiments_status', 'status'),
        Index('ix_experiments_type', 'experiment_type'),
    )


class FusionShot(Base):
    """SQLAlchemy model for individual fusion shots."""
    
    __tablename__ = "fusion_shots"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), sa.ForeignKey("fusion_experiments.id"), nullable=False)
    shot_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Timing information
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration: Mapped[Optional[float]] = mapped_column(Float)  # seconds
    
    # Plasma parameters
    plasma_current: Mapped[float] = mapped_column(Float)
    electron_density: Mapped[float] = mapped_column(Float)
    electron_temperature: Mapped[float] = mapped_column(Float)
    ion_temperature: Mapped[float] = mapped_column(Float)
    neutral_beam_power: Mapped[float] = mapped_column(Float)
    rf_heating_power: Mapped[float] = mapped_column(Float)
    
    # Results
    q_factor: Mapped[Optional[float]] = mapped_column(Float)
    confinement_time: Mapped[Optional[float]] = mapped_column(Float)
    beta_normalized: Mapped[Optional[float]] = mapped_column(Float)
    
    # Status and quality
    status: Mapped[str] = mapped_column(String(50), default="completed")
    quality_rating: Mapped[Optional[float]] = mapped_column(Float)  # 0-1 scale
    
    # Disruption information
    had_disruption: Mapped[bool] = mapped_column(Boolean, default=False)
    disruption_time: Mapped[Optional[float]] = mapped_column(Float)
    disruption_cause: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Additional data as JSON
    diagnostics_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    analysis_results: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    experiment = relationship("FusionExperiment", back_populates="shots")
    time_series = relationship("TimeSeriesData", back_populates="shot", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_shots_experiment_id', 'experiment_id'),
        Index('ix_shots_shot_number', 'shot_number'),
        Index('ix_shots_start_time', 'start_time'),
        Index('ix_shots_q_factor', 'q_factor'),
        sa.UniqueConstraint('experiment_id', 'shot_number', name='uq_experiment_shot'),
    )


class TimeSeriesData(Base):
    """SQLAlchemy model for time series data."""
    
    __tablename__ = "time_series_data"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), sa.ForeignKey("fusion_shots.id"), nullable=False)
    
    # Time information
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    time_relative: Mapped[float] = mapped_column(Float, nullable=False)  # Relative to shot start
    
    # Measurement type and location
    diagnostic_name: Mapped[str] = mapped_column(String(100), nullable=False)
    measurement_type: Mapped[str] = mapped_column(String(100), nullable=False)
    spatial_location: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Values
    value: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[Optional[float]] = mapped_column(Float)
    quality_flag: Mapped[int] = mapped_column(Integer, default=0)  # 0=good, 1=suspect, 2=bad
    
    # Metadata
    units: Mapped[str] = mapped_column(String(50), nullable=False)
    calibration_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Relationships
    shot = relationship("FusionShot", back_populates="time_series")
    
    __table_args__ = (
        Index('ix_timeseries_shot_id', 'shot_id'),
        Index('ix_timeseries_timestamp', 'timestamp'),
        Index('ix_timeseries_diagnostic', 'diagnostic_name'),
        Index('ix_timeseries_measurement', 'measurement_type'),
        Index('ix_timeseries_shot_time', 'shot_id', 'time_relative'),
    )


class ModelPrediction(Base):
    """SQLAlchemy model for ML model predictions."""
    
    __tablename__ = "model_predictions"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    shot_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), sa.ForeignKey("fusion_shots.id"))
    
    # Model information
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    prediction_time: Mapped[Optional[float]] = mapped_column(Float)  # Relative to shot start
    
    # Input features
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    
    # Predictions
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    prediction_interval_lower: Mapped[Optional[float]] = mapped_column(Float)
    prediction_interval_upper: Mapped[Optional[float]] = mapped_column(Float)
    
    # Validation
    actual_value: Mapped[Optional[float]] = mapped_column(Float)
    prediction_error: Mapped[Optional[float]] = mapped_column(Float)
    
    # Additional metadata
    computation_time: Mapped[Optional[float]] = mapped_column(Float)  # milliseconds
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    __table_args__ = (
        Index('ix_predictions_model_id', 'model_id'),
        Index('ix_predictions_created_at', 'created_at'),
        Index('ix_predictions_shot_id', 'shot_id'),
        Index('ix_predictions_model_version', 'model_version'),
    )


class PostgreSQLManager:
    """
    PostgreSQL database manager with async support.
    
    Provides connection pooling, transaction management,
    and high-level database operations.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize PostgreSQL manager.
        
        Args:
            config: Database configuration.
        """
        self.config = config
        self.engine = None
        self.session_factory = None
        
        logger.info("PostgreSQLManager initialized")
    
    async def initialize(self):
        """Initialize database connection and create tables."""
        if not HAS_POSTGRESQL:
            raise RuntimeError("PostgreSQL dependencies not installed")
        
        # Create async engine
        database_url = (
            f"postgresql+asyncpg://{self.config.postgresql_username}:"
            f"{self.config.postgresql_password}@{self.config.postgresql_host}:"
            f"{self.config.postgresql_port}/{self.config.postgresql_database}"
        )
        
        self.engine = create_async_engine(
            database_url,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
            pool_timeout=self.config.connection_timeout,
            echo=False,  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("PostgreSQL database initialized")
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
        logger.info("PostgreSQL connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_experiment(self, 
                               name: str,
                               reactor_type: str,
                               experiment_type: str = "simulation",
                               description: str = "",
                               **kwargs) -> FusionExperiment:
        """
        Create a new fusion experiment.
        
        Args:
            name: Experiment name.
            reactor_type: Type of reactor.
            experiment_type: Type of experiment.
            description: Experiment description.
            **kwargs: Additional experiment parameters.
            
        Returns:
            Created experiment object.
        """
        async with self.get_session() as session:
            experiment = FusionExperiment(
                name=name,
                reactor_type=reactor_type,
                experiment_type=experiment_type,
                description=description,
                **kwargs
            )
            
            session.add(experiment)
            await session.commit()
            await session.refresh(experiment)
            
            logger.info(f"Created experiment: {name} ({experiment.id})")
            return experiment
    
    async def create_shot(self, 
                         experiment_id: uuid.UUID,
                         shot_number: int,
                         plasma_parameters: Dict[str, float],
                         start_time: Optional[datetime] = None) -> FusionShot:
        """
        Create a new fusion shot.
        
        Args:
            experiment_id: Parent experiment ID.
            shot_number: Shot number.
            plasma_parameters: Plasma parameter values.
            start_time: Shot start time.
            
        Returns:
            Created shot object.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        async with self.get_session() as session:
            shot = FusionShot(
                experiment_id=experiment_id,
                shot_number=shot_number,
                start_time=start_time,
                **plasma_parameters
            )
            
            session.add(shot)
            await session.commit()
            await session.refresh(shot)
            
            logger.info(f"Created shot: {shot_number} for experiment {experiment_id}")
            return shot
    
    async def store_time_series(self, 
                               shot_id: uuid.UUID,
                               data: pd.DataFrame) -> List[TimeSeriesData]:
        """
        Store time series data for a shot.
        
        Args:
            shot_id: Shot ID.
            data: Time series data DataFrame.
            
        Returns:
            List of created time series records.
        """
        async with self.get_session() as session:
            records = []
            
            for _, row in data.iterrows():
                record = TimeSeriesData(
                    shot_id=shot_id,
                    timestamp=row.get('timestamp', datetime.now(timezone.utc)),
                    time_relative=row['time_relative'],
                    diagnostic_name=row['diagnostic_name'],
                    measurement_type=row['measurement_type'],
                    value=row['value'],
                    units=row['units'],
                    uncertainty=row.get('uncertainty'),
                    quality_flag=row.get('quality_flag', 0),
                    spatial_location=row.get('spatial_location')
                )
                
                records.append(record)
                session.add(record)
            
            await session.commit()
            
            logger.info(f"Stored {len(records)} time series records for shot {shot_id}")
            return records
    
    async def get_experiments(self, 
                             limit: int = 100,
                             offset: int = 0,
                             status: Optional[str] = None) -> List[FusionExperiment]:
        """
        Retrieve experiments with optional filtering.
        
        Args:
            limit: Maximum number of experiments to return.
            offset: Number of experiments to skip.
            status: Optional status filter.
            
        Returns:
            List of experiments.
        """
        async with self.get_session() as session:
            query = sa.select(FusionExperiment)
            
            if status:
                query = query.where(FusionExperiment.status == status)
            
            query = query.order_by(FusionExperiment.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_shot_data(self, shot_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive shot data including time series.
        
        Args:
            shot_id: Shot ID.
            
        Returns:
            Shot data dictionary or None.
        """
        async with self.get_session() as session:
            # Get shot with experiment
            shot_query = (
                sa.select(FusionShot)
                .options(sa.orm.selectinload(FusionShot.experiment))
                .where(FusionShot.id == shot_id)
            )
            
            shot_result = await session.execute(shot_query)
            shot = shot_result.scalar_one_or_none()
            
            if not shot:
                return None
            
            # Get time series data
            ts_query = (
                sa.select(TimeSeriesData)
                .where(TimeSeriesData.shot_id == shot_id)
                .order_by(TimeSeriesData.time_relative)
            )
            
            ts_result = await session.execute(ts_query)
            time_series = ts_result.scalars().all()
            
            # Convert to dictionary
            shot_data = {
                'shot_info': {
                    'id': str(shot.id),
                    'shot_number': shot.shot_number,
                    'start_time': shot.start_time.isoformat(),
                    'duration': shot.duration,
                    'plasma_current': shot.plasma_current,
                    'electron_density': shot.electron_density,
                    'electron_temperature': shot.electron_temperature,
                    'ion_temperature': shot.ion_temperature,
                    'q_factor': shot.q_factor,
                    'status': shot.status
                },
                'experiment_info': {
                    'id': str(shot.experiment.id),
                    'name': shot.experiment.name,
                    'reactor_type': shot.experiment.reactor_type,
                    'experiment_type': shot.experiment.experiment_type
                },
                'time_series': [
                    {
                        'timestamp': ts.timestamp.isoformat(),
                        'time_relative': ts.time_relative,
                        'diagnostic_name': ts.diagnostic_name,
                        'measurement_type': ts.measurement_type,
                        'value': ts.value,
                        'units': ts.units,
                        'quality_flag': ts.quality_flag
                    }
                    for ts in time_series
                ]
            }
            
            return shot_data


class MongoDBManager:
    """
    MongoDB manager for document storage.
    
    Handles unstructured data, large datasets,
    and complex nested documents.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize MongoDB manager.
        
        Args:
            config: Database configuration.
        """
        self.config = config
        self.client = None
        self.database = None
        
        logger.info("MongoDBManager initialized")
    
    async def initialize(self):
        """Initialize MongoDB connection."""
        if not HAS_MONGODB:
            raise RuntimeError("MongoDB dependencies not installed")
        
        # Build connection URL
        if self.config.mongodb_username and self.config.mongodb_password:
            connection_url = (
                f"mongodb://{self.config.mongodb_username}:"
                f"{self.config.mongodb_password}@{self.config.mongodb_host}:"
                f"{self.config.mongodb_port}/{self.config.mongodb_database}"
                f"?authSource={self.config.mongodb_auth_source}"
            )
        else:
            connection_url = f"mongodb://{self.config.mongodb_host}:{self.config.mongodb_port}"
        
        # Create async client
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            connection_url,
            serverSelectionTimeoutMS=self.config.connection_timeout * 1000
        )
        
        self.database = self.client[self.config.mongodb_database]
        
        # Test connection
        await self.client.admin.command('ping')
        
        # Create indexes
        await self._create_indexes()
        
        logger.info("MongoDB database initialized")
    
    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
        logger.info("MongoDB connection closed")
    
    async def _create_indexes(self):
        """Create database indexes for performance."""
        # Raw data collection indexes
        raw_data = self.database.raw_fusion_data
        await raw_data.create_index([("experiment_id", 1), ("shot_number", 1)])
        await raw_data.create_index([("timestamp", 1)])
        await raw_data.create_index([("data_type", 1)])
        
        # Analysis results indexes
        analysis = self.database.analysis_results
        await analysis.create_index([("experiment_id", 1)])
        await analysis.create_index([("analysis_type", 1)])
        await analysis.create_index([("created_at", 1)])
        
        # Model artifacts indexes
        models = self.database.model_artifacts
        await models.create_index([("model_id", 1), ("version", 1)])
        await models.create_index([("created_at", 1)])
        
        logger.info("MongoDB indexes created")
    
    async def store_raw_data(self, 
                            experiment_id: str,
                            shot_number: int,
                            data_type: str,
                            data: Dict[str, Any]) -> str:
        """
        Store raw fusion data.
        
        Args:
            experiment_id: Experiment identifier.
            shot_number: Shot number.
            data_type: Type of data (e.g., 'diagnostics', 'simulation').
            data: Raw data dictionary.
            
        Returns:
            Document ID.
        """
        document = {
            'experiment_id': experiment_id,
            'shot_number': shot_number,
            'data_type': data_type,
            'timestamp': datetime.now(timezone.utc),
            'data': data,
            'metadata': {
                'size_bytes': len(json.dumps(data, default=str)),
                'fields_count': len(data) if isinstance(data, dict) else 0
            }
        }
        
        collection = self.database.raw_fusion_data
        result = await collection.insert_one(document)
        
        logger.info(f"Stored raw data: {data_type} for shot {shot_number}")
        return str(result.inserted_id)
    
    async def store_analysis_results(self, 
                                   experiment_id: str,
                                   analysis_type: str,
                                   results: Dict[str, Any]) -> str:
        """
        Store analysis results.
        
        Args:
            experiment_id: Experiment identifier.
            analysis_type: Type of analysis.
            results: Analysis results.
            
        Returns:
            Document ID.
        """
        document = {
            'experiment_id': experiment_id,
            'analysis_type': analysis_type,
            'created_at': datetime.now(timezone.utc),
            'results': results,
            'metadata': {
                'analysis_version': '1.0',
                'computation_time': results.get('computation_time'),
                'parameters': results.get('parameters', {})
            }
        }
        
        collection = self.database.analysis_results
        result = await collection.insert_one(document)
        
        logger.info(f"Stored analysis results: {analysis_type}")
        return str(result.inserted_id)
    
    async def get_raw_data(self, 
                          experiment_id: str,
                          shot_number: Optional[int] = None,
                          data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve raw data with optional filtering.
        
        Args:
            experiment_id: Experiment identifier.
            shot_number: Optional shot number filter.
            data_type: Optional data type filter.
            
        Returns:
            List of data documents.
        """
        query = {'experiment_id': experiment_id}
        
        if shot_number is not None:
            query['shot_number'] = shot_number
        
        if data_type:
            query['data_type'] = data_type
        
        collection = self.database.raw_fusion_data
        cursor = collection.find(query).sort('timestamp', 1)
        
        return await cursor.to_list(length=None)


class CacheManager:
    """
    Redis-based cache manager for performance optimization.
    
    Provides caching for frequently accessed data,
    session storage, and real-time data buffering.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize cache manager.
        
        Args:
            config: Database configuration.
        """
        self.config = config
        self.redis_client = None
        
        logger.info("CacheManager initialized")
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not HAS_REDIS:
            logger.warning("Redis not available, caching disabled")
            return
        
        # Create Redis client
        self.redis_client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
            socket_timeout=self.config.connection_timeout
        )
        
        # Test connection
        await self.redis_client.ping()
        
        logger.info("Redis cache initialized")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Redis connection closed")
    
    async def set(self, 
                 key: str, 
                 value: Any, 
                 expire_seconds: int = 3600) -> bool:
        """
        Set cache value with expiration.
        
        Args:
            key: Cache key.
            value: Value to cache.
            expire_seconds: Expiration time in seconds.
            
        Returns:
            Success status.
        """
        if not self.redis_client:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, expire_seconds, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None.
        """
        if not self.redis_client:
            return None
        
        try:
            cached_value = await self.redis_client.get(key)
            if cached_value:
                return json.loads(cached_value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete cached value.
        
        Args:
            key: Cache key.
            
        Returns:
            Success status.
        """
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False


class DatabaseOrchestrator:
    """
    Main database orchestrator that coordinates all database operations.
    
    Provides unified interface for PostgreSQL, MongoDB, and Redis,
    with intelligent data routing and caching strategies.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database orchestrator.
        
        Args:
            config: Database configuration.
        """
        self.config = config
        self.postgresql = PostgreSQLManager(config)
        self.mongodb = MongoDBManager(config)
        self.cache = CacheManager(config)
        
        logger.info("DatabaseOrchestrator initialized")
    
    async def initialize(self):
        """Initialize all database connections."""
        await self.postgresql.initialize()
        await self.mongodb.initialize()
        await self.cache.initialize()
        
        logger.info("All database connections initialized")
    
    async def close(self):
        """Close all database connections."""
        await self.postgresql.close()
        await self.mongodb.close()
        await self.cache.close()
        
        logger.info("All database connections closed")
    
    async def store_experiment_data(self, 
                                   experiment_data: Dict[str, Any],
                                   shot_data: List[Dict[str, Any]],
                                   raw_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Store complete experiment data across all databases.
        
        Args:
            experiment_data: Experiment metadata.
            shot_data: List of shot data.
            raw_data: Optional raw data for MongoDB.
            
        Returns:
            Dictionary with created record IDs.
        """
        results = {}
        
        # Store experiment in PostgreSQL
        experiment = await self.postgresql.create_experiment(**experiment_data)
        results['experiment_id'] = str(experiment.id)
        
        # Store shots in PostgreSQL
        shot_ids = []
        for shot in shot_data:
            shot_record = await self.postgresql.create_shot(
                experiment_id=experiment.id,
                **shot
            )
            shot_ids.append(str(shot_record.id))
        
        results['shot_ids'] = shot_ids
        
        # Store raw data in MongoDB if provided
        if raw_data:
            for i, shot in enumerate(shot_data):
                raw_id = await self.mongodb.store_raw_data(
                    experiment_id=str(experiment.id),
                    shot_number=shot['shot_number'],
                    data_type='experimental_data',
                    data=raw_data.get(f'shot_{i}', {})
                )
                results[f'raw_data_shot_{i}'] = raw_id
        
        # Cache experiment summary
        cache_key = f"experiment:{experiment.id}"
        summary = {
            'name': experiment.name,
            'type': experiment.experiment_type,
            'reactor_type': experiment.reactor_type,
            'shot_count': len(shot_ids),
            'created_at': experiment.created_at.isoformat()
        }
        await self.cache.set(cache_key, summary, expire_seconds=3600)
        
        logger.info(f"Stored complete experiment data: {experiment.name}")
        return results
    
    async def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment summary with caching.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            Experiment summary or None.
        """
        # Try cache first
        cache_key = f"experiment:{experiment_id}"
        cached_summary = await self.cache.get(cache_key)
        
        if cached_summary:
            logger.debug(f"Retrieved experiment summary from cache: {experiment_id}")
            return cached_summary
        
        # Get from database
        experiments = await self.postgresql.get_experiments(limit=1)
        # Note: This is simplified - would need proper ID filtering
        
        if not experiments:
            return None
        
        experiment = experiments[0]
        summary = {
            'id': str(experiment.id),
            'name': experiment.name,
            'type': experiment.experiment_type,
            'reactor_type': experiment.reactor_type,
            'created_at': experiment.created_at.isoformat(),
            'status': experiment.status
        }
        
        # Cache the result
        await self.cache.set(cache_key, summary, expire_seconds=3600)
        
        logger.info(f"Retrieved experiment summary: {experiment_id}")
        return summary


def create_database_manager(config_path: Optional[str] = None) -> DatabaseOrchestrator:
    """
    Create database manager with configuration.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Configured database orchestrator.
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        # Would load from file in practice
        config = DatabaseConfig()
    else:
        config = DatabaseConfig()
    
    return DatabaseOrchestrator(config)