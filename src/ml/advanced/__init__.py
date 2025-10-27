"""
Advanced machine learning models for nuclear fusion analysis.

This module implements state-of-the-art ML models including:
- Ensemble methods (XGBoost, LightGBM, CatBoost)
- Deep learning models (transformers, autoencoders)
- Time series forecasting models
- Anomaly detection models
- Multi-modal learning approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
import warnings

# Core ML libraries
try:
    import sklearn
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        VotingRegressor, StackingRegressor, BaggingRegressor
    )
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Advanced ensemble methods
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Time series
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    HAS_ANOMALY = True
except ImportError:
    HAS_ANOMALY = False

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Advanced model configuration."""
    
    # General settings
    model_type: str = "ensemble"
    random_state: int = 42
    n_jobs: int = -1
    
    # Ensemble settings
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    use_voting: bool = True
    use_stacking: bool = True
    
    # Deep learning settings
    use_deep_learning: bool = True
    use_transformers: bool = False
    epochs: int = 100
    batch_size: int = 32
    
    # Optimization settings
    hyperparameter_tuning: bool = True
    cv_folds: int = 5
    optimization_metric: str = "r2"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    training_time: float
    prediction_time: float
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'mape': self.mape,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_size_mb': self.model_size_mb
        }


class BaseAdvancedModel(ABC):
    """Base class for advanced ML models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: Model configuration.
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance_ = None
        self.training_history = None
        self.is_trained = False
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Features.
            y: True targets.
            
        Returns:
            Model metrics.
        """
        import time
        
        # Prediction timing
        start_time = time.time()
        y_pred = self.predict(X)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-8))) * 100
        
        # Model size estimation
        model_size_mb = self._estimate_model_size()
        
        return ModelMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            training_time=getattr(self, '_training_time', 0.0),
            prediction_time=prediction_time,
            model_size_mb=model_size_mb
        )
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        try:
            # Save model temporarily to estimate size
            temp_path = "/tmp/temp_model.pkl"
            joblib.dump(self.model, temp_path)
            size_bytes = Path(temp_path).stat().st_size
            Path(temp_path).unlink()  # Clean up
            return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_importance_ = model_data.get('feature_importance')
        self.training_history = model_data.get('training_history')
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {path}")


class EnsembleModel(BaseAdvancedModel):
    """
    Advanced ensemble model combining multiple algorithms.
    
    Combines XGBoost, LightGBM, CatBoost, and traditional models.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize ensemble model."""
        super().__init__(config)
        self.base_models = {}
        self.ensemble_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train ensemble model."""
        import time
        start_time = time.time()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Prepare base models
        base_models = []
        
        # XGBoost
        if self.config.use_xgboost and HAS_XGB:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            base_models.append(('xgb', xgb_model))
            self.base_models['xgb'] = xgb_model
        
        # LightGBM
        if self.config.use_lightgbm and HAS_LGB:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=-1
            )
            base_models.append(('lgb', lgb_model))
            self.base_models['lgb'] = lgb_model
        
        # CatBoost
        if self.config.use_catboost and HAS_CATBOOST:
            cat_model = cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=False
            )
            base_models.append(('cat', cat_model))
            self.base_models['cat'] = cat_model
        
        # Traditional models
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        base_models.append(('rf', rf_model))
        self.base_models['rf'] = rf_model
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state
        )
        base_models.append(('gb', gb_model))
        self.base_models['gb'] = gb_model
        
        # Create ensemble
        if self.config.use_stacking:
            # Stacking ensemble
            from sklearn.linear_model import Ridge
            meta_model = Ridge(alpha=1.0)
            
            self.ensemble_model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs
            )
        else:
            # Voting ensemble
            self.ensemble_model = VotingRegressor(
                estimators=base_models,
                n_jobs=self.config.n_jobs
            )
        
        # Train ensemble
        self.ensemble_model.fit(X_scaled, y)
        self.model = self.ensemble_model
        
        # Calculate feature importance (average across models)
        self._calculate_feature_importance(X.shape[1])
        
        self._training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Ensemble model trained in {self._training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with ensemble."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict(X_scaled)
    
    def _calculate_feature_importance(self, n_features: int):
        """Calculate average feature importance across base models."""
        importances = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        
        if importances:
            self.feature_importance_ = np.mean(importances, axis=0)
        else:
            self.feature_importance_ = np.ones(n_features) / n_features


class TransformerModel(BaseAdvancedModel):
    """
    Transformer-based model for sequential fusion data.
    
    Uses attention mechanisms for temporal dependencies.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize transformer model."""
        super().__init__(config)
        
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available for transformer model")
        
        self.sequence_length = 50
        self.d_model = 128
        self.num_heads = 8
        self.num_layers = 4
        self.dff = 512
        self.dropout_rate = 0.1
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train transformer model."""
        import time
        start_time = time.time()
        
        # Prepare sequential data
        X_seq, y_seq = self._prepare_sequential_data(X, y)
        
        if X_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequential_data(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build transformer model
        self.model = self._build_transformer(X_seq.shape[-1])
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.training_history = history.history
        self._training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Transformer model trained in {self._training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with transformer."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_seq, _ = self._prepare_sequential_data(X, np.zeros(X.shape[0]))
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
    def _prepare_sequential_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sequential modeling."""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def _build_transformer(self, input_dim: int) -> tf.keras.Model:
        """Build transformer architecture."""
        inputs = layers.Input(shape=(self.sequence_length, input_dim))
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(self.dff, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.dff // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        
        # Create positional encoding
        pos_encoding = self._get_positional_encoding(self.sequence_length, d_model)
        
        return x + pos_encoding[:seq_len, :]
    
    def _get_positional_encoding(self, seq_len: int, d_model: int):
        """Generate positional encoding."""
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.cast(tf.range(0, d_model, 2), tf.float32) * 
                         -(np.log(10000.0) / d_model))
        
        pe = tf.zeros((seq_len, d_model))
        pe = tf.tensor_scatter_nd_update(
            pe, 
            tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
            tf.sin(position * div_term)
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
            tf.cos(position * div_term)
        )
        
        return pe[tf.newaxis, ...]
    
    def _transformer_block(self, x):
        """Single transformer block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        
        # Add & norm
        x1 = layers.Add()([x, attention_output])
        x1 = layers.LayerNormalization()(x1)
        
        # Feed forward
        ffn_output = layers.Dense(self.dff, activation='relu')(x1)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        
        # Add & norm
        x2 = layers.Add()([x1, ffn_output])
        x2 = layers.LayerNormalization()(x2)
        
        return x2


class TimeSeriesModel(BaseAdvancedModel):
    """
    Advanced time series model for fusion parameter forecasting.
    
    Combines statistical and ML approaches.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize time series model."""
        super().__init__(config)
        
        self.statistical_models = {}
        self.ml_model = None
        self.seasonal_period = 24  # Assume hourly data with daily seasonality
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train time series model."""
        import time
        start_time = time.time()
        
        # Convert to time series
        ts_data = pd.Series(y)
        
        # Fit statistical models
        self._fit_statistical_models(ts_data)
        
        # Prepare features for ML model
        features = self._create_time_features(X, y)
        
        # Fit ML model
        if HAS_XGB:
            self.ml_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=self.config.random_state
            )
            self.ml_model.fit(features, y)
        
        self.model = {
            'statistical': self.statistical_models,
            'ml': self.ml_model
        }
        
        self._training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Time series model trained in {self._training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make time series predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # ML predictions
        features = self._create_time_features(X, np.zeros(X.shape[0]))
        ml_predictions = self.ml_model.predict(features)
        
        # For now, return ML predictions
        # In practice, would ensemble with statistical models
        return ml_predictions
    
    def _fit_statistical_models(self, ts_data: pd.Series):
        """Fit statistical time series models."""
        try:
            # ARIMA model
            if HAS_STATSMODELS:
                arima_model = ARIMA(ts_data, order=(2, 1, 2))
                self.statistical_models['arima'] = arima_model.fit()
                
                # Exponential smoothing
                exp_smooth = ExponentialSmoothing(
                    ts_data,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.seasonal_period
                )
                self.statistical_models['exp_smooth'] = exp_smooth.fit()
        
        except Exception as e:
            logger.warning(f"Statistical model fitting failed: {e}")
    
    def _create_time_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create time-based features."""
        n_samples = X.shape[0]
        
        # Basic lag features
        lag_features = []
        for lag in [1, 2, 3, 6, 12, 24]:
            if lag < len(y):
                lag_values = np.concatenate([np.full(lag, y[0]), y[:-lag]])
                lag_features.append(lag_values)
        
        # Rolling statistics
        rolling_features = []
        window_sizes = [3, 6, 12, 24]
        
        for window in window_sizes:
            if window < len(y):
                # Rolling mean
                rolling_mean = pd.Series(y).rolling(window=window, min_periods=1).mean().values
                rolling_features.append(rolling_mean)
                
                # Rolling std
                rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().fillna(0).values
                rolling_features.append(rolling_std)
        
        # Combine original features with time features
        time_features = np.column_stack([X] + lag_features + rolling_features)
        
        return time_features


class AnomalyDetectionModel(BaseAdvancedModel):
    """
    Advanced anomaly detection for fusion system monitoring.
    
    Combines multiple anomaly detection techniques.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize anomaly detection model."""
        super().__init__(config)
        
        if not HAS_ANOMALY:
            raise RuntimeError("Anomaly detection libraries not available")
        
        self.models = {}
        self.ensemble_weights = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train anomaly detection models."""
        import time
        start_time = time.time()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        # One-Class SVM
        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.models['one_class_svm'].fit(X_scaled)
        
        # Local Outlier Factor
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
            n_jobs=self.config.n_jobs
        )
        self.models['lof'].fit(X_scaled)
        
        # Autoencoder for deep anomaly detection
        if HAS_TF:
            self.models['autoencoder'] = self._build_autoencoder(X_scaled.shape[1])
            self.models['autoencoder'].fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        
        # Set ensemble weights
        self.ensemble_weights = {
            'isolation_forest': 0.3,
            'one_class_svm': 0.3,
            'lof': 0.3,
            'autoencoder': 0.1 if HAS_TF else 0.0
        }
        
        self.model = self.models
        self._training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Anomaly detection models trained in {self._training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        
        # Isolation Forest (returns -1 for outliers, 1 for inliers)
        if_pred = self.models['isolation_forest'].decision_function(X_scaled)
        predictions['isolation_forest'] = (if_pred - if_pred.min()) / (if_pred.max() - if_pred.min())
        
        # One-Class SVM
        svm_pred = self.models['one_class_svm'].decision_function(X_scaled)
        predictions['one_class_svm'] = (svm_pred - svm_pred.min()) / (svm_pred.max() - svm_pred.min())
        
        # Local Outlier Factor
        lof_pred = self.models['lof'].decision_function(X_scaled)
        predictions['lof'] = (lof_pred - lof_pred.min()) / (lof_pred.max() - lof_pred.min())
        
        # Autoencoder reconstruction error
        if 'autoencoder' in self.models:
            reconstructed = self.models['autoencoder'].predict(X_scaled, verbose=0)
            reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
            predictions['autoencoder'] = (reconstruction_error - reconstruction_error.min()) / \
                                       (reconstruction_error.max() - reconstruction_error.min())
        
        # Ensemble prediction
        ensemble_scores = np.zeros(X.shape[0])
        for model_name, scores in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            ensemble_scores += weight * scores
        
        return ensemble_scores
    
    def _build_autoencoder(self, input_dim: int) -> tf.keras.Model:
        """Build autoencoder for anomaly detection."""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = tf.keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder


class AdvancedModelSuite:
    """
    Comprehensive suite of advanced ML models.
    
    Provides unified interface for multiple model types.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model suite.
        
        Args:
            config: Model configuration.
        """
        self.config = config
        self.models: Dict[str, BaseAdvancedModel] = {}
        self.best_model: Optional[BaseAdvancedModel] = None
        self.model_comparison: Optional[pd.DataFrame] = None
        
        logger.info("AdvancedModelSuite initialized")
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train all available models.
        
        Args:
            X: Training features.
            y: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        """
        logger.info("Training all advanced models...")
        
        # Ensemble model
        try:
            ensemble_model = EnsembleModel(self.config)
            ensemble_model.fit(X, y, X_val, y_val)
            self.models['ensemble'] = ensemble_model
            logger.info("✓ Ensemble model trained")
        except Exception as e:
            logger.error(f"Ensemble model training failed: {e}")
        
        # Transformer model
        if self.config.use_transformers and HAS_TF:
            try:
                transformer_model = TransformerModel(self.config)
                transformer_model.fit(X, y, X_val, y_val)
                self.models['transformer'] = transformer_model
                logger.info("✓ Transformer model trained")
            except Exception as e:
                logger.error(f"Transformer model training failed: {e}")
        
        # Time series model
        try:
            ts_model = TimeSeriesModel(self.config)
            ts_model.fit(X, y, X_val, y_val)
            self.models['time_series'] = ts_model
            logger.info("✓ Time series model trained")
        except Exception as e:
            logger.error(f"Time series model training failed: {e}")
        
        # Anomaly detection model
        try:
            anomaly_model = AnomalyDetectionModel(self.config)
            anomaly_model.fit(X, y, X_val, y_val)
            self.models['anomaly_detection'] = anomaly_model
            logger.info("✓ Anomaly detection model trained")
        except Exception as e:
            logger.error(f"Anomaly detection model training failed: {e}")
        
        logger.info(f"Training completed. {len(self.models)} models available.")
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Comparison DataFrame.
        """
        if not self.models:
            raise ValueError("No models trained")
        
        results = []
        
        for name, model in self.models.items():
            try:
                if name == 'anomaly_detection':
                    # Skip evaluation for anomaly detection
                    continue
                
                metrics = model.evaluate(X_test, y_test)
                
                result = {
                    'model': name,
                    **metrics.to_dict()
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Model {name} evaluation failed: {e}")
        
        self.model_comparison = pd.DataFrame(results)
        
        # Find best model based on R²
        if not self.model_comparison.empty:
            best_idx = self.model_comparison['r2'].idxmax()
            best_model_name = self.model_comparison.loc[best_idx, 'model']
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Best model: {best_model_name} (R² = {self.model_comparison.loc[best_idx, 'r2']:.4f})")
        
        return self.model_comparison
    
    def get_best_model(self) -> Optional[BaseAdvancedModel]:
        """Get the best performing model."""
        return self.best_model
    
    def predict_ensemble(self, X: np.ndarray, use_weights: bool = True) -> np.ndarray:
        """
        Make ensemble predictions from all regression models.
        
        Args:
            X: Features for prediction.
            use_weights: Whether to use performance-based weights.
            
        Returns:
            Ensemble predictions.
        """
        if not self.models:
            raise ValueError("No models trained")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if name == 'anomaly_detection':
                continue  # Skip anomaly detection for regression
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
                
                # Use R² as weight if available
                if use_weights and self.model_comparison is not None:
                    model_row = self.model_comparison[self.model_comparison['model'] == name]
                    if not model_row.empty:
                        weight = max(0, model_row['r2'].iloc[0])
                        weights.append(weight)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)
                    
            except Exception as e:
                logger.error(f"Prediction failed for model {name}: {e}")
        
        if not predictions:
            raise ValueError("No successful predictions")
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def save_suite(self, base_path: str):
        """Save all models in the suite."""
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = base_path / f"{name}_model.pkl"
            model.save_model(str(model_path))
        
        # Save comparison results
        if self.model_comparison is not None:
            comparison_path = base_path / "model_comparison.csv"
            self.model_comparison.to_csv(comparison_path, index=False)
        
        logger.info(f"Model suite saved to {base_path}")
    
    def load_suite(self, base_path: str):
        """Load all models in the suite."""
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Model suite path not found: {base_path}")
        
        # Load models
        for model_file in base_path.glob("*_model.pkl"):
            name = model_file.stem.replace("_model", "")
            
            # Create appropriate model instance
            if name == 'ensemble':
                model = EnsembleModel(self.config)
            elif name == 'transformer':
                model = TransformerModel(self.config)
            elif name == 'time_series':
                model = TimeSeriesModel(self.config)
            elif name == 'anomaly_detection':
                model = AnomalyDetectionModel(self.config)
            else:
                continue
            
            model.load_model(str(model_file))
            self.models[name] = model
        
        # Load comparison results
        comparison_path = base_path / "model_comparison.csv"
        if comparison_path.exists():
            self.model_comparison = pd.read_csv(comparison_path)
            
            # Set best model
            if not self.model_comparison.empty:
                best_idx = self.model_comparison['r2'].idxmax()
                best_model_name = self.model_comparison.loc[best_idx, 'model']
                if best_model_name in self.models:
                    self.best_model = self.models[best_model_name]
        
        logger.info(f"Model suite loaded from {base_path}")


def create_advanced_model_suite(config: Optional[ModelConfig] = None) -> AdvancedModelSuite:
    """
    Create configured advanced model suite.
    
    Args:
        config: Model configuration. Uses default if None.
        
    Returns:
        Configured model suite.
    """
    if config is None:
        config = ModelConfig()
    
    return AdvancedModelSuite(config)