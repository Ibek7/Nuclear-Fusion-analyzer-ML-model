"""
Database models for Nuclear Fusion Analyzer.

This module defines SQLAlchemy ORM models for PostgreSQL
and Pydantic models for data validation and serialization.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# SQLAlchemy imports with fallback
try:
    import sqlalchemy as sa
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text, Index
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Pydantic imports with fallback
try:
    from pydantic import BaseModel, Field, validator, ConfigDict
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ShotStatus(str, Enum):
    """Shot status enumeration."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    DISRUPTED = "disrupted"
    FAILED = "failed"


class QualityFlag(int, Enum):
    """Data quality flag enumeration."""
    GOOD = 0
    SUSPECT = 1
    BAD = 2


# SQLAlchemy Models (if available)
if HAS_SQLALCHEMY:
    
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
        status: Mapped[str] = mapped_column(String(50), default=ExperimentStatus.PLANNED.value)
        
        # Reactor configuration
        reactor_type: Mapped[str] = mapped_column(String(100), nullable=False)
        major_radius: Mapped[Optional[float]] = mapped_column(Float)
        minor_radius: Mapped[Optional[float]] = mapped_column(Float)
        magnetic_field: Mapped[Optional[float]] = mapped_column(Float)
        
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
        plasma_current: Mapped[Optional[float]] = mapped_column(Float)
        electron_density: Mapped[Optional[float]] = mapped_column(Float)
        electron_temperature: Mapped[Optional[float]] = mapped_column(Float)
        ion_temperature: Mapped[Optional[float]] = mapped_column(Float)
        neutral_beam_power: Mapped[Optional[float]] = mapped_column(Float)
        rf_heating_power: Mapped[Optional[float]] = mapped_column(Float)
        
        # Results
        q_factor: Mapped[Optional[float]] = mapped_column(Float)
        confinement_time: Mapped[Optional[float]] = mapped_column(Float)
        beta_normalized: Mapped[Optional[float]] = mapped_column(Float)
        
        # Status and quality
        status: Mapped[str] = mapped_column(String(50), default=ShotStatus.COMPLETED.value)
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
        quality_flag: Mapped[int] = mapped_column(Integer, default=QualityFlag.GOOD.value)
        
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

else:
    # Placeholder classes if SQLAlchemy is not available
    class Base:
        pass
    
    class FusionExperiment:
        pass
    
    class FusionShot:
        pass
    
    class TimeSeriesData:
        pass
    
    class ModelPrediction:
        pass


# Pydantic Models (if available)
if HAS_PYDANTIC:
    
    class ExperimentCreate(BaseModel):
        """Pydantic model for creating experiments."""
        
        name: str = Field(..., min_length=1, max_length=255)
        description: Optional[str] = Field(None, max_length=2000)
        experiment_type: str = Field(..., min_length=1, max_length=100)
        reactor_type: str = Field(..., min_length=1, max_length=100)
        major_radius: Optional[float] = Field(None, gt=0)
        minor_radius: Optional[float] = Field(None, gt=0)
        magnetic_field: Optional[float] = Field(None, gt=0)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True
        )
    
    
    class ExperimentResponse(BaseModel):
        """Pydantic model for experiment responses."""
        
        id: str
        name: str
        description: Optional[str]
        experiment_type: str
        reactor_type: str
        status: str
        created_at: datetime
        updated_at: datetime
        major_radius: Optional[float]
        minor_radius: Optional[float]
        magnetic_field: Optional[float]
        metadata: Dict[str, Any]
        
        model_config = ConfigDict(
            from_attributes=True
        )
    
    
    class ShotCreate(BaseModel):
        """Pydantic model for creating shots."""
        
        experiment_id: str
        shot_number: int = Field(..., gt=0)
        start_time: Optional[datetime] = None
        plasma_current: Optional[float] = Field(None, ge=0)
        electron_density: Optional[float] = Field(None, ge=0)
        electron_temperature: Optional[float] = Field(None, ge=0)
        ion_temperature: Optional[float] = Field(None, ge=0)
        neutral_beam_power: Optional[float] = Field(None, ge=0)
        rf_heating_power: Optional[float] = Field(None, ge=0)
        
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True
        )
        
        @validator('start_time', pre=True, always=True)
        def set_start_time(cls, v):
            return v or datetime.now(timezone.utc)
    
    
    class ShotResponse(BaseModel):
        """Pydantic model for shot responses."""
        
        id: str
        experiment_id: str
        shot_number: int
        start_time: datetime
        end_time: Optional[datetime]
        duration: Optional[float]
        status: str
        plasma_current: Optional[float]
        electron_density: Optional[float]
        electron_temperature: Optional[float]
        ion_temperature: Optional[float]
        q_factor: Optional[float]
        confinement_time: Optional[float]
        beta_normalized: Optional[float]
        had_disruption: bool
        quality_rating: Optional[float]
        
        model_config = ConfigDict(
            from_attributes=True
        )
    
    
    class TimeSeriesDataCreate(BaseModel):
        """Pydantic model for creating time series data."""
        
        shot_id: str
        timestamp: Optional[datetime] = None
        time_relative: float
        diagnostic_name: str = Field(..., min_length=1, max_length=100)
        measurement_type: str = Field(..., min_length=1, max_length=100)
        spatial_location: Optional[str] = Field(None, max_length=255)
        value: float
        uncertainty: Optional[float] = Field(None, ge=0)
        quality_flag: int = Field(default=QualityFlag.GOOD.value, ge=0, le=2)
        units: str = Field(..., min_length=1, max_length=50)
        calibration_id: Optional[str] = Field(None, max_length=100)
        
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True
        )
        
        @validator('timestamp', pre=True, always=True)
        def set_timestamp(cls, v):
            return v or datetime.now(timezone.utc)
    
    
    class TimeSeriesDataResponse(BaseModel):
        """Pydantic model for time series data responses."""
        
        id: str
        shot_id: str
        timestamp: datetime
        time_relative: float
        diagnostic_name: str
        measurement_type: str
        spatial_location: Optional[str]
        value: float
        uncertainty: Optional[float]
        quality_flag: int
        units: str
        calibration_id: Optional[str]
        
        model_config = ConfigDict(
            from_attributes=True
        )
    
    
    class PredictionCreate(BaseModel):
        """Pydantic model for creating predictions."""
        
        shot_id: Optional[str] = None
        model_id: str = Field(..., min_length=1, max_length=255)
        model_version: str = Field(..., min_length=1, max_length=100)
        prediction_type: str = Field(..., min_length=1, max_length=100)
        prediction_time: Optional[float] = None
        input_features: Dict[str, Any]
        predicted_value: float
        confidence_score: Optional[float] = Field(None, ge=0, le=1)
        prediction_interval_lower: Optional[float] = None
        prediction_interval_upper: Optional[float] = None
        computation_time: Optional[float] = Field(None, ge=0)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True
        )
    
    
    class PredictionResponse(BaseModel):
        """Pydantic model for prediction responses."""
        
        id: str
        shot_id: Optional[str]
        model_id: str
        model_version: str
        prediction_type: str
        created_at: datetime
        prediction_time: Optional[float]
        input_features: Dict[str, Any]
        predicted_value: float
        confidence_score: Optional[float]
        prediction_interval_lower: Optional[float]
        prediction_interval_upper: Optional[float]
        actual_value: Optional[float]
        prediction_error: Optional[float]
        computation_time: Optional[float]
        metadata: Dict[str, Any]
        
        model_config = ConfigDict(
            from_attributes=True
        )
    
    
    class DatabaseStats(BaseModel):
        """Pydantic model for database statistics."""
        
        total_experiments: int
        total_shots: int
        total_time_series_points: int
        total_predictions: int
        active_experiments: int
        recent_shots: int
        data_quality_summary: Dict[str, int]
        storage_usage: Dict[str, str]
        
        model_config = ConfigDict(
            validate_assignment=True
        )

else:
    # Placeholder classes if Pydantic is not available
    class ExperimentCreate:
        pass
    
    class ExperimentResponse:
        pass
    
    class ShotCreate:
        pass
    
    class ShotResponse:
        pass
    
    class TimeSeriesDataCreate:
        pass
    
    class TimeSeriesDataResponse:
        pass
    
    class PredictionCreate:
        pass
    
    class PredictionResponse:
        pass
    
    class DatabaseStats:
        pass


def get_model_by_name(model_name: str):
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model.
        
    Returns:
        Model class or None.
    """
    models = {
        'FusionExperiment': FusionExperiment,
        'FusionShot': FusionShot,
        'TimeSeriesData': TimeSeriesData,
        'ModelPrediction': ModelPrediction
    }
    
    return models.get(model_name)


def validate_model_data(model_class, data: Dict[str, Any]) -> bool:
    """
    Validate data against a Pydantic model.
    
    Args:
        model_class: Pydantic model class.
        data: Data to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not HAS_PYDANTIC:
        return True  # Skip validation if Pydantic not available
    
    try:
        model_class(**data)
        return True
    except Exception:
        return False