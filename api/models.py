"""
Pydantic models for Nuclear Fusion Analyzer API.

Defines request and response schemas for the REST API endpoints.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class PlasmaParameters(BaseModel):
    """Model for plasma parameters input."""
    
    magnetic_field: float = Field(..., ge=0.1, le=15.0, description="Magnetic field strength in Tesla")
    plasma_current: float = Field(..., ge=0.1, le=50.0, description="Plasma current in MA")
    electron_density: float = Field(..., ge=1e18, le=1e22, description="Electron density in m^-3")
    ion_temperature: float = Field(..., ge=0.1, le=100.0, description="Ion temperature in keV")
    electron_temperature: float = Field(..., ge=0.1, le=100.0, description="Electron temperature in keV")
    
    @validator('electron_density')
    def validate_density(cls, v):
        if not (1e18 <= v <= 1e22):
            raise ValueError('Electron density must be between 1e18 and 1e22 m^-3')
        return v


class HeatingParameters(BaseModel):
    """Model for heating system parameters."""
    
    neutral_beam_power: float = Field(0.0, ge=0.0, le=200.0, description="Neutral beam power in MW")
    rf_heating_power: float = Field(0.0, ge=0.0, le=100.0, description="RF heating power in MW")
    ohmic_heating_power: float = Field(0.0, ge=0.0, le=50.0, description="Ohmic heating power in MW")
    heating_efficiency: float = Field(0.8, ge=0.3, le=1.0, description="Heating efficiency")


class FuelParameters(BaseModel):
    """Model for fuel system parameters."""
    
    fuel_density: float = Field(..., ge=1e18, le=1e21, description="Fuel density in m^-3")
    deuterium_fraction: float = Field(0.5, ge=0.0, le=1.0, description="Deuterium fraction")
    tritium_fraction: float = Field(0.5, ge=0.0, le=1.0, description="Tritium fraction")
    impurity_concentration: float = Field(0.02, ge=0.0, le=0.2, description="Impurity concentration")
    
    @validator('tritium_fraction')
    def validate_fuel_fractions(cls, v, values):
        if 'deuterium_fraction' in values:
            if values['deuterium_fraction'] + v > 1.1:  # Allow small numerical errors
                raise ValueError('Deuterium and tritium fractions cannot sum to more than 1')
        return v


class FusionPredictionRequest(BaseModel):
    """Model for fusion prediction request."""
    
    plasma: PlasmaParameters
    heating: Optional[HeatingParameters] = None
    fuel: Optional[FuelParameters] = None
    
    class Config:
        schema_extra = {
            "example": {
                "plasma": {
                    "magnetic_field": 5.3,
                    "plasma_current": 15.0,
                    "electron_density": 1.0e20,
                    "ion_temperature": 20.0,
                    "electron_temperature": 15.0
                },
                "heating": {
                    "neutral_beam_power": 50.0,
                    "rf_heating_power": 30.0,
                    "ohmic_heating_power": 10.0,
                    "heating_efficiency": 0.85
                },
                "fuel": {
                    "fuel_density": 5.0e19,
                    "deuterium_fraction": 0.5,
                    "tritium_fraction": 0.5,
                    "impurity_concentration": 0.02
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""
    
    predictions: List[FusionPredictionRequest] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Model for prediction response."""
    
    q_factor: float = Field(..., description="Predicted Q factor")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    performance_category: str = Field(..., description="Performance category")
    
    class Config:
        schema_extra = {
            "example": {
                "q_factor": 1.25,
                "confidence": 0.92,
                "performance_category": "Breakeven"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    
    predictions: List[PredictionResponse]
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")


class AnomalyDetectionRequest(BaseModel):
    """Model for anomaly detection request."""
    
    data: Dict[str, float] = Field(..., description="Fusion parameters to check for anomalies")
    detectors: Optional[List[str]] = Field(None, description="Specific detectors to use")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "magnetic_field": 5.3,
                    "plasma_current": 15.0,
                    "electron_density": 1.0e20,
                    "ion_temperature": 20.0,
                    "q_factor": 1.2
                },
                "detectors": ["isolation_forest", "physics_based"]
            }
        }


class AnomalyDetectionResponse(BaseModel):
    """Model for anomaly detection response."""
    
    is_anomaly: bool = Field(..., description="Whether data is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score")
    detector_results: Dict[str, Any] = Field(..., description="Results from individual detectors")
    anomaly_type: Optional[str] = Field(None, description="Type of anomaly detected")


class ModelTrainingRequest(BaseModel):
    """Model for training request."""
    
    model_name: str = Field(..., description="Name of the model to train")
    n_samples: int = Field(10000, ge=100, le=100000, description="Number of training samples")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters")


class ModelTrainingResponse(BaseModel):
    """Model for training response."""
    
    success: bool = Field(..., description="Training success status")
    model_name: str = Field(..., description="Name of trained model")
    performance_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    training_time: float = Field(..., description="Training time in seconds")


class DataGenerationRequest(BaseModel):
    """Model for data generation request."""
    
    n_samples: int = Field(1000, ge=10, le=50000, description="Number of samples to generate")
    anomaly_rate: float = Field(0.05, ge=0.0, le=0.5, description="Fraction of anomalous samples")
    include_time_series: bool = Field(False, description="Whether to include time series data")
    
    class Config:
        schema_extra = {
            "example": {
                "n_samples": 5000,
                "anomaly_rate": 0.1,
                "include_time_series": false
            }
        }


class DataGenerationResponse(BaseModel):
    """Model for data generation response."""
    
    success: bool = Field(..., description="Generation success status")
    n_samples: int = Field(..., description="Number of samples generated")
    columns: List[str] = Field(..., description="Column names in generated data")
    statistics: Dict[str, Any] = Field(..., description="Basic statistics of generated data")


class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="Currently loaded models")
    uptime: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": {
                    "field": "magnetic_field",
                    "issue": "Value must be positive"
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    trained: bool = Field(..., description="Whether model is trained")
    performance: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class ModelsListResponse(BaseModel):
    """Response for models list endpoint."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_models: int = Field(..., description="Total number of models")


# Utility functions for model conversion
def numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj