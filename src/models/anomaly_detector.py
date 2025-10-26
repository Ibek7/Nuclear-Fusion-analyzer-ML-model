"""
Anomaly detection models for nuclear fusion monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Anomaly detection will be limited.")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Deep anomaly detection disabled.")
    TENSORFLOW_AVAILABLE = False


class FusionAnomalyDetector:
    """
    Anomaly detection system for nuclear fusion operations.
    Detects disruptions, equipment failures, and unusual plasma behavior.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.trained_models = {}
        self.scaler = None
        self.feature_names = None
        
        if SKLEARN_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize anomaly detection models."""
        contamination = self.config.get('contamination', 0.1)
        
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=contamination,
                gamma='scale'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=contamination,
                n_jobs=-1
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            )
        }
    
    def create_autoencoder(self, input_dim: int) -> 'keras.Model':
        """
        Create autoencoder for anomaly detection.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled autoencoder model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for anomaly detection.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data array
        """
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        processed_data = data[numerical_cols].copy()
        
        # Handle missing values
        processed_data = processed_data.fillna(processed_data.median())
        
        # Scale features
        if self.scaler is None:
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(processed_data)
            else:
                # Manual scaling
                means = processed_data.mean()
                stds = processed_data.std()
                scaled_data = (processed_data - means) / stds
                scaled_data = scaled_data.values
        else:
            if SKLEARN_AVAILABLE:
                scaled_data = self.scaler.transform(processed_data)
            else:
                scaled_data = processed_data.values
        
        self.feature_names = numerical_cols.tolist()
        return scaled_data
    
    def train_isolation_forest(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Train Isolation Forest for anomaly detection.
        
        Args:
            X: Training data
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        model = self.models['isolation_forest']
        model.fit(X)
        
        # Predict on training data
        predictions = model.predict(X)
        scores = model.decision_function(X)
        
        self.trained_models['isolation_forest'] = model
        
        return {
            'model_name': 'isolation_forest',
            'anomaly_count': np.sum(predictions == -1),
            'normal_count': np.sum(predictions == 1),
            'anomaly_scores': scores,
            'threshold': np.percentile(scores, 10)
        }
    
    def train_autoencoder(self, X: np.ndarray, 
                         validation_split: float = 0.2,
                         epochs: int = 100) -> Dict[str, Any]:
        """
        Train autoencoder for anomaly detection.
        
        Args:
            X: Training data (normal samples only)
            validation_split: Fraction for validation
            epochs: Training epochs
            
        Returns:
            Training results
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Create autoencoder
        autoencoder = self.create_autoencoder(X.shape[1])
        
        # Train autoencoder
        history = autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=20, restore_best_weights=True
                )
            ]
        )
        
        # Calculate reconstruction errors
        reconstructions = autoencoder.predict(X)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Set threshold based on training data
        threshold = np.percentile(reconstruction_errors, 95)
        
        self.trained_models['autoencoder'] = autoencoder
        
        return {
            'model_name': 'autoencoder',
            'reconstruction_errors': reconstruction_errors,
            'threshold': threshold,
            'training_history': history.history
        }
    
    def detect_plasma_disruptions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect plasma disruption events.
        
        Args:
            data: Fusion data with plasma parameters
            
        Returns:
            Data with disruption predictions
        """
        results = data.copy()
        
        # Define disruption indicators
        disruption_indicators = []
        
        # Beta limit violation
        if all(col in data.columns for col in ['beta_plasma', 'safety_factor']):
            troyon_limit = 0.028 * data['safety_factor']
            beta_violation = data['beta_plasma'] > troyon_limit
            disruption_indicators.append(beta_violation)
        
        # Low safety factor
        if 'safety_factor' in data.columns:
            low_q = data['safety_factor'] < 2.0
            disruption_indicators.append(low_q)
        
        # High impurity concentration
        if 'impurity_concentration' in data.columns:
            high_impurity = data['impurity_concentration'] > 0.1
            disruption_indicators.append(high_impurity)
        
        # Sudden temperature drop
        if 'plasma_temperature' in data.columns:
            temp_gradient = np.gradient(data['plasma_temperature'])
            sudden_cooling = temp_gradient < -1e7  # Large negative gradient
            disruption_indicators.append(sudden_cooling)
        
        # Combine indicators
        if disruption_indicators:
            disruption_risk = np.sum(disruption_indicators, axis=0) / len(disruption_indicators)
            results['disruption_risk'] = disruption_risk
            results['disruption_predicted'] = disruption_risk > 0.5
        else:
            results['disruption_risk'] = 0.0
            results['disruption_predicted'] = False
        
        return results
    
    def detect_equipment_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect equipment health anomalies.
        
        Args:
            data: Data with system health indicators
            
        Returns:
            Data with equipment anomaly predictions
        """
        results = data.copy()
        
        # Health score threshold
        health_threshold = 0.8
        
        health_columns = [col for col in data.columns if 'health' in col]
        
        if health_columns:
            # Individual system anomalies
            for col in health_columns:
                system_name = col.replace('_health', '')
                results[f'{system_name}_anomaly'] = data[col] < health_threshold
            
            # Overall system health
            overall_health = data[health_columns].mean(axis=1)
            results['equipment_anomaly'] = overall_health < health_threshold
            results['equipment_health_score'] = overall_health
        else:
            results['equipment_anomaly'] = False
            results['equipment_health_score'] = 1.0
        
        return results
    
    def detect_operational_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect operational parameter anomalies.
        
        Args:
            data: Operational data
            
        Returns:
            Data with operational anomaly predictions
        """
        results = data.copy()
        
        anomaly_flags = []
        
        # Q factor anomaly (too low for given conditions)
        if 'q_factor' in data.columns:
            q_anomaly = data['q_factor'] < 0.1
            anomaly_flags.append(q_anomaly)
            results['q_factor_anomaly'] = q_anomaly
        
        # Power balance anomaly
        if all(col in data.columns for col in ['fusion_power', 'total_heating_power']):
            power_ratio = data['fusion_power'] / (data['total_heating_power'] + 1e-6)
            power_anomaly = power_ratio > 2.0  # Unrealistic power gain
            anomaly_flags.append(power_anomaly)
            results['power_balance_anomaly'] = power_anomaly
        
        # Confinement time anomaly
        if 'confinement_time' in data.columns:
            confinement_anomaly = (data['confinement_time'] < 0.05) | \
                                (data['confinement_time'] > 5.0)
            anomaly_flags.append(confinement_anomaly)
            results['confinement_anomaly'] = confinement_anomaly
        
        # Combine operational anomalies
        if anomaly_flags:
            operational_anomaly = np.any(anomaly_flags, axis=0)
            results['operational_anomaly'] = operational_anomaly
        else:
            results['operational_anomaly'] = False
        
        return results
    
    def comprehensive_anomaly_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive anomaly detection.
        
        Args:
            data: Input fusion data
            
        Returns:
            Comprehensive anomaly detection results
        """
        results = {}
        
        # Preprocess data
        X = self.preprocess_data(data)
        
        # Train and apply multiple anomaly detection methods
        if SKLEARN_AVAILABLE:
            # Isolation Forest
            try:
                iso_results = self.train_isolation_forest(X)
                results['isolation_forest'] = iso_results
            except Exception as e:
                print(f"Isolation Forest failed: {e}")
        
        # Autoencoder
        if TENSORFLOW_AVAILABLE:
            try:
                ae_results = self.train_autoencoder(X)
                results['autoencoder'] = ae_results
            except Exception as e:
                print(f"Autoencoder failed: {e}")
        
        # Domain-specific anomaly detection
        disruption_results = self.detect_plasma_disruptions(data)
        equipment_results = self.detect_equipment_anomalies(data)
        operational_results = self.detect_operational_anomalies(data)
        
        # Combine results
        combined_data = data.copy()
        
        # Add all anomaly flags
        anomaly_columns = []
        
        for df in [disruption_results, equipment_results, operational_results]:
            for col in df.columns:
                if 'anomaly' in col or 'predicted' in col:
                    combined_data[col] = df[col]
                    anomaly_columns.append(col)
        
        # Overall anomaly score
        if anomaly_columns:
            anomaly_scores = combined_data[anomaly_columns].astype(int)
            combined_data['overall_anomaly_score'] = anomaly_scores.sum(axis=1)
            combined_data['is_anomaly'] = combined_data['overall_anomaly_score'] > 0
        
        results['combined_data'] = combined_data
        results['anomaly_summary'] = self._generate_anomaly_summary(combined_data)
        
        return results
    
    def _generate_anomaly_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of detected anomalies.
        
        Args:
            data: Data with anomaly predictions
            
        Returns:
            Anomaly summary dictionary
        """
        summary = {
            'total_samples': len(data),
            'anomaly_counts': {},
            'anomaly_rates': {}
        }
        
        anomaly_columns = [col for col in data.columns 
                          if 'anomaly' in col or 'predicted' in col]
        
        for col in anomaly_columns:
            if col in data.columns:
                count = data[col].sum()
                rate = count / len(data)
                summary['anomaly_counts'][col] = int(count)
                summary['anomaly_rates'][col] = float(rate)
        
        return summary
    
    def predict_anomalies(self, data: pd.DataFrame, 
                         model_name: str = 'isolation_forest') -> np.ndarray:
        """
        Predict anomalies using trained model.
        
        Args:
            data: Input data
            model_name: Name of model to use
            
        Returns:
            Anomaly predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        X = self.preprocess_data(data)
        model = self.trained_models[model_name]
        
        if model_name == 'autoencoder' and TENSORFLOW_AVAILABLE:
            reconstructions = model.predict(X)
            reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
            # Use stored threshold
            threshold = self.config.get('autoencoder_threshold', 
                                      np.percentile(reconstruction_errors, 95))
            return reconstruction_errors > threshold
        else:
            predictions = model.predict(X)
            return predictions == -1  # Anomalies are labeled as -1