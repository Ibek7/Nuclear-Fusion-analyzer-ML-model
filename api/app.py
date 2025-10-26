"""
FastAPI application for Nuclear Fusion Analyzer.

Provides REST API endpoints for fusion prediction, anomaly detection,
model training, and data generation.
"""

import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.models import (
    FusionPredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, AnomalyDetectionRequest, AnomalyDetectionResponse,
    ModelTrainingRequest, ModelTrainingResponse, DataGenerationRequest, 
    DataGenerationResponse, HealthCheckResponse, ErrorResponse, ModelsListResponse,
    ModelInfo, numpy_to_python
)
from src.data.generator import FusionDataGenerator
from src.data.processor import FusionDataProcessor
from src.models.fusion_predictor import FusionPredictor
from src.models.anomaly_detector import FusionAnomalyDetector
from src.utils.config_manager import get_global_config, get_config_for_environment


# Initialize FastAPI app
app = FastAPI(
    title="Nuclear Fusion Analyzer API",
    description="REST API for nuclear fusion prediction and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.config = get_config_for_environment('development')
        self.predictor = FusionPredictor()
        self.anomaly_detector = FusionAnomalyDetector()
        self.data_generator = FusionDataGenerator()
        self.data_processor = FusionDataProcessor()
        self.trained_models = set()
        
    def get_uptime(self) -> float:
        return time.time() - self.start_time

# Initialize app state
app_state = AppState()


# Helper functions
def convert_request_to_dataframe(request: FusionPredictionRequest) -> pd.DataFrame:
    """Convert prediction request to DataFrame format."""
    data = {}
    
    # Add plasma parameters
    for field, value in request.plasma.dict().items():
        data[field] = [value]
    
    # Add heating parameters if provided
    if request.heating:
        for field, value in request.heating.dict().items():
            data[field] = [value]
    
    # Add fuel parameters if provided  
    if request.fuel:
        for field, value in request.fuel.dict().items():
            data[field] = [value]
    
    return pd.DataFrame(data)


def categorize_q_factor(q_factor: float) -> str:
    """Categorize Q factor into performance levels."""
    if q_factor < 0.1:
        return "Poor"
    elif q_factor < 1.0:
        return "Sub-critical"
    elif q_factor < 5.0:
        return "Breakeven"
    elif q_factor < 10.0:
        return "Ignition"
    else:
        return "High Performance"


# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "details": {"traceback": traceback.format_exc()}
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API health status."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=list(app_state.trained_models),
        uptime=app_state.get_uptime()
    )


# Models endpoints
@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List available models and their status."""
    models = []
    
    # Available model types
    model_types = {
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'support_vector': 'Support Vector Regression',
        'deep_learning': 'Deep Neural Network'
    }
    
    for model_name, model_type in model_types.items():
        is_trained = model_name in app_state.trained_models
        performance = None
        
        if is_trained and model_name in app_state.predictor.trained_models:
            # Get model performance if available
            performance = {"status": "trained"}
        
        models.append(ModelInfo(
            name=model_name,
            type=model_type,
            trained=is_trained,
            performance=performance,
            last_updated=datetime.now().isoformat() if is_trained else None
        ))
    
    return ModelsListResponse(
        models=models,
        total_models=len(models)
    )


@app.post("/models/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a specific model."""
    try:
        start_time = time.time()
        
        # Generate training data
        data = app_state.data_generator.generate_dataset(n_samples=request.n_samples)
        
        # Preprocess data
        processed_data = app_state.data_processor.preprocess_pipeline(
            data, target_column='q_factor'
        )
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train model with custom hyperparameters if provided
        if request.hyperparameters:
            # Update model hyperparameters
            if request.model_name in app_state.predictor.models:
                model = app_state.predictor.models[request.model_name]
                model.set_params(**request.hyperparameters)
        
        # Train the model
        results = app_state.predictor.train_model(
            request.model_name, X_train, y_train, X_val, y_val
        )
        
        app_state.trained_models.add(request.model_name)
        training_time = time.time() - start_time
        
        return ModelTrainingResponse(
            success=True,
            model_name=request.model_name,
            performance_metrics={
                "train_r2": numpy_to_python(results['train_r2']),
                "val_r2": numpy_to_python(results['val_r2']),
                "train_rmse": numpy_to_python(results['train_rmse']),
                "val_rmse": numpy_to_python(results['val_rmse'])
            },
            training_time=training_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_fusion(request: FusionPredictionRequest):
    """Predict Q factor for given fusion parameters."""
    try:
        # Check if any model is trained
        if not app_state.trained_models:
            raise HTTPException(
                status_code=400, 
                detail="No models are trained. Please train a model first."
            )
        
        # Convert request to DataFrame
        df = convert_request_to_dataframe(request)
        
        # Use the first available trained model
        model_name = next(iter(app_state.trained_models))
        
        # Make prediction
        prediction = app_state.predictor.predict(df.values, model_name)
        q_factor = float(prediction[0])
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.6, 1.0 - abs(q_factor - 1.0) * 0.2))
        
        return PredictionResponse(
            q_factor=q_factor,
            confidence=confidence,
            performance_category=categorize_q_factor(q_factor)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Perform batch predictions."""
    try:
        if not app_state.trained_models:
            raise HTTPException(
                status_code=400,
                detail="No models are trained. Please train a model first."
            )
        
        predictions = []
        q_factors = []
        
        # Process each prediction request
        for pred_request in request.predictions:
            df = convert_request_to_dataframe(pred_request)
            model_name = next(iter(app_state.trained_models))
            
            prediction = app_state.predictor.predict(df.values, model_name)
            q_factor = float(prediction[0])
            q_factors.append(q_factor)
            
            confidence = min(0.95, max(0.6, 1.0 - abs(q_factor - 1.0) * 0.2))
            
            predictions.append(PredictionResponse(
                q_factor=q_factor,
                confidence=confidence,
                performance_category=categorize_q_factor(q_factor)
            ))
        
        # Calculate summary statistics
        summary = {
            "total_predictions": len(predictions),
            "average_q_factor": float(np.mean(q_factors)),
            "min_q_factor": float(np.min(q_factors)),
            "max_q_factor": float(np.max(q_factors)),
            "breakeven_count": sum(1 for q in q_factors if q >= 1.0),
            "ignition_count": sum(1 for q in q_factors if q >= 5.0)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Anomaly detection endpoints
@app.post("/anomaly/detect", response_model=AnomalyDetectionResponse)
async def detect_anomaly(request: AnomalyDetectionRequest):
    """Detect anomalies in fusion parameters."""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame([request.data])
        
        # Run physics-based anomaly detection
        physics_anomalies = app_state.anomaly_detector.physics_based_anomaly_detection(df)
        
        # Run disruption detection if applicable
        disruption_anomalies = app_state.anomaly_detector.detect_plasma_disruptions(df)
        
        # Check for equipment failures
        equipment_failures = app_state.anomaly_detector.detect_equipment_failures(df)
        
        # Combine results
        is_anomaly = bool(physics_anomalies[0] or disruption_anomalies[0] or equipment_failures[0])
        
        # Calculate anomaly score
        anomaly_score = 0.0
        if physics_anomalies[0]:
            anomaly_score += 0.4
        if disruption_anomalies[0]:
            anomaly_score += 0.4
        if equipment_failures[0]:
            anomaly_score += 0.2
        
        # Determine anomaly type
        anomaly_type = None
        if is_anomaly:
            if physics_anomalies[0]:
                anomaly_type = "Physics Violation"
            elif disruption_anomalies[0]:
                anomaly_type = "Plasma Disruption"
            elif equipment_failures[0]:
                anomaly_type = "Equipment Failure"
        
        detector_results = {
            "physics_based": bool(physics_anomalies[0]),
            "disruption_detection": bool(disruption_anomalies[0]),
            "equipment_failure": bool(equipment_failures[0])
        }
        
        return AnomalyDetectionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            detector_results=detector_results,
            anomaly_type=anomaly_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Data generation endpoints
@app.post("/data/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest):
    """Generate synthetic fusion data."""
    try:
        # Generate dataset
        data = app_state.data_generator.generate_dataset(
            n_samples=request.n_samples
        )
        
        # Inject anomalies if requested
        if request.anomaly_rate > 0:
            data = app_state.data_generator._inject_anomalies(
                data, anomaly_rate=request.anomaly_rate
            )
        
        # Calculate basic statistics
        statistics = {
            "mean_q_factor": float(data['q_factor'].mean()),
            "std_q_factor": float(data['q_factor'].std()),
            "min_q_factor": float(data['q_factor'].min()),
            "max_q_factor": float(data['q_factor'].max()),
            "breakeven_fraction": float((data['q_factor'] >= 1.0).mean())
        }
        
        return DataGenerationResponse(
            success=True,
            n_samples=len(data),
            columns=list(data.columns),
            statistics=statistics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/config")
async def get_config():
    """Get current configuration."""
    try:
        return app_state.config.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main application entry point
if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )