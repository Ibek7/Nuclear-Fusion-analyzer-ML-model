"""
Advanced anomaly detection system for nuclear fusion reactors.

This module provides comprehensive anomaly detection capabilities
including statistical methods, machine learning approaches, ensemble
detection, real-time monitoring, and fusion-specific anomaly patterns.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

try:
    # Statistical libraries
    from scipy import stats
    from scipy.special import erf
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # Machine learning for anomaly detection
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.covariance import EllipticEnvelope
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    # Advanced anomaly detection
    import pyod
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.iforest import IForest
    from pyod.models.hbos import HBOS
    from pyod.models.feature_bagging import FeatureBagging
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies in fusion systems."""
    POINT = "point"           # Single point anomaly
    CONTEXTUAL = "contextual" # Contextual anomaly
    COLLECTIVE = "collective" # Collective/pattern anomaly
    TREND = "trend"          # Trend anomaly
    SEASONAL = "seasonal"    # Seasonal anomaly


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    
    timestamps: np.ndarray
    anomaly_scores: np.ndarray
    anomaly_labels: np.ndarray  # 1 for anomaly, 0 for normal
    anomaly_types: List[AnomalyType]
    severity_levels: List[AnomalySeverity]
    confidence_scores: np.ndarray
    method_name: str
    threshold: float
    metadata: Dict[str, Any]


@dataclass
class FusionAnomalyPattern:
    """Fusion-specific anomaly pattern definition."""
    
    name: str
    description: str
    parameters: List[str]
    detection_rules: Dict[str, Any]
    severity: AnomalySeverity
    response_actions: List[str]


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection methods.
    
    Implements various statistical approaches for anomaly detection
    including Z-score, modified Z-score, IQR, and Grubbs test.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize statistical detector.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config or {}
        self.threshold = self.config.get('threshold', 3.0)
        logger.info("StatisticalAnomalyDetector initialized")
    
    def z_score_detection(self, data: np.ndarray) -> AnomalyResult:
        """Detect anomalies using Z-score method."""
        mean = np.mean(data)
        std = np.std(data)
        
        z_scores = np.abs((data - mean) / (std + 1e-8))
        anomaly_labels = (z_scores > self.threshold).astype(int)
        
        return AnomalyResult(
            timestamps=np.arange(len(data)),
            anomaly_scores=z_scores,
            anomaly_labels=anomaly_labels,
            anomaly_types=[AnomalyType.POINT] * len(data),
            severity_levels=[self._score_to_severity(score) for score in z_scores],
            confidence_scores=np.minimum(z_scores / self.threshold, 1.0),
            method_name='z_score',
            threshold=self.threshold,
            metadata={'mean': mean, 'std': std}
        )
    
    def modified_z_score_detection(self, data: np.ndarray) -> AnomalyResult:
        """Detect anomalies using modified Z-score (median-based)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        
        modified_z_scores = 0.6745 * (data - median) / (mad + 1e-8)
        modified_z_scores = np.abs(modified_z_scores)
        
        threshold = self.config.get('modified_z_threshold', 3.5)
        anomaly_labels = (modified_z_scores > threshold).astype(int)
        
        return AnomalyResult(
            timestamps=np.arange(len(data)),
            anomaly_scores=modified_z_scores,
            anomaly_labels=anomaly_labels,
            anomaly_types=[AnomalyType.POINT] * len(data),
            severity_levels=[self._score_to_severity(score) for score in modified_z_scores],
            confidence_scores=np.minimum(modified_z_scores / threshold, 1.0),
            method_name='modified_z_score',
            threshold=threshold,
            metadata={'median': median, 'mad': mad}
        )
    
    def iqr_detection(self, data: np.ndarray) -> AnomalyResult:
        """Detect anomalies using Interquartile Range method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        multiplier = self.config.get('iqr_multiplier', 1.5)
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Calculate anomaly scores
        anomaly_scores = np.maximum(
            (lower_bound - data) / (iqr + 1e-8),
            (data - upper_bound) / (iqr + 1e-8)
        )
        anomaly_scores = np.maximum(anomaly_scores, 0)
        
        anomaly_labels = ((data < lower_bound) | (data > upper_bound)).astype(int)
        
        return AnomalyResult(
            timestamps=np.arange(len(data)),
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_labels,
            anomaly_types=[AnomalyType.POINT] * len(data),
            severity_levels=[self._score_to_severity(score) for score in anomaly_scores],
            confidence_scores=np.minimum(anomaly_scores, 1.0),
            method_name='iqr',
            threshold=0.0,
            metadata={'q1': q1, 'q3': q3, 'iqr': iqr, 'bounds': (lower_bound, upper_bound)}
        )
    
    def grubbs_test(self, data: np.ndarray) -> AnomalyResult:
        """Detect anomalies using Grubbs test for outliers."""
        if not HAS_SCIPY:
            raise ImportError("SciPy is required for Grubbs test")
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        # Calculate Grubbs statistic for each point
        grubbs_stats = np.abs(data - mean) / (std + 1e-8)
        
        # Critical value for Grubbs test
        alpha = self.config.get('grubbs_alpha', 0.05)
        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
        
        anomaly_labels = (grubbs_stats > grubbs_critical).astype(int)
        
        return AnomalyResult(
            timestamps=np.arange(len(data)),
            anomaly_scores=grubbs_stats,
            anomaly_labels=anomaly_labels,
            anomaly_types=[AnomalyType.POINT] * len(data),
            severity_levels=[self._score_to_severity(score) for score in grubbs_stats],
            confidence_scores=np.minimum(grubbs_stats / grubbs_critical, 1.0),
            method_name='grubbs',
            threshold=grubbs_critical,
            metadata={'critical_value': grubbs_critical, 'alpha': alpha}
        )
    
    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert anomaly score to severity level."""
        if score < 2.0:
            return AnomalySeverity.LOW
        elif score < 3.0:
            return AnomalySeverity.MEDIUM
        elif score < 4.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


class MLAnomalyDetector:
    """
    Machine learning-based anomaly detection.
    
    Implements various ML algorithms for anomaly detection including
    Isolation Forest, One-Class SVM, Local Outlier Factor, and ensemble methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ML detector.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.contamination = self.config.get('contamination', 0.1)
        
        logger.info("MLAnomalyDetector initialized")
    
    def fit_isolation_forest(self, X: np.ndarray) -> str:
        """Fit Isolation Forest model."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for Isolation Forest")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=self.config.get('n_estimators', 100)
        )
        model.fit(X_scaled)
        
        model_id = 'isolation_forest'
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        
        return model_id
    
    def fit_one_class_svm(self, X: np.ndarray) -> str:
        """Fit One-Class SVM model."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for One-Class SVM")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model = OneClassSVM(
            kernel=self.config.get('svm_kernel', 'rbf'),
            gamma=self.config.get('svm_gamma', 'scale'),
            nu=self.contamination
        )
        model.fit(X_scaled)
        
        model_id = 'one_class_svm'
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        
        return model_id
    
    def fit_local_outlier_factor(self, X: np.ndarray) -> str:
        """Fit Local Outlier Factor model."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for LOF")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model = LocalOutlierFactor(
            n_neighbors=self.config.get('lof_neighbors', 20),
            contamination=self.contamination,
            novelty=True  # For prediction on new data
        )
        model.fit(X_scaled)
        
        model_id = 'local_outlier_factor'
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        
        return model_id
    
    def fit_ensemble_detector(self, X: np.ndarray) -> str:
        """Fit ensemble of multiple detectors."""
        model_ids = []
        
        # Fit individual models
        try:
            model_ids.append(self.fit_isolation_forest(X))
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
        
        try:
            model_ids.append(self.fit_one_class_svm(X))
        except Exception as e:
            logger.warning(f"One-Class SVM failed: {e}")
        
        try:
            model_ids.append(self.fit_local_outlier_factor(X))
        except Exception as e:
            logger.warning(f"LOF failed: {e}")
        
        if not model_ids:
            raise ValueError("No models could be fitted for ensemble")
        
        ensemble_id = 'ensemble'
        self.models[ensemble_id] = model_ids
        
        return ensemble_id
    
    def detect_anomalies(self, X: np.ndarray, model_id: str) -> AnomalyResult:
        """Detect anomalies using fitted model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not fitted")
        
        if model_id == 'ensemble':
            return self._ensemble_detect(X)
        
        model = self.models[model_id]
        scaler = self.scalers[model_id]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions and scores
        if hasattr(model, 'decision_function'):
            anomaly_scores = -model.decision_function(X_scaled)  # More negative = more anomalous
            anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-8)
        else:
            anomaly_scores = model.score_samples(X_scaled)
            anomaly_scores = -anomaly_scores  # Convert to positive anomaly scores
            anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-8)
        
        anomaly_labels = model.predict(X_scaled)
        anomaly_labels = (anomaly_labels == -1).astype(int)  # Convert -1 to 1 for anomaly
        
        return AnomalyResult(
            timestamps=np.arange(len(X)),
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_labels,
            anomaly_types=[AnomalyType.POINT] * len(X),
            severity_levels=[self._score_to_severity(score) for score in anomaly_scores],
            confidence_scores=anomaly_scores,
            method_name=model_id,
            threshold=0.5,
            metadata={'contamination': self.contamination}
        )
    
    def _ensemble_detect(self, X: np.ndarray) -> AnomalyResult:
        """Ensemble anomaly detection."""
        model_ids = self.models['ensemble']
        
        all_scores = []
        all_labels = []
        
        for model_id in model_ids:
            try:
                result = self.detect_anomalies(X, model_id)
                all_scores.append(result.anomaly_scores)
                all_labels.append(result.anomaly_labels)
            except Exception as e:
                logger.warning(f"Model {model_id} failed in ensemble: {e}")
        
        if not all_scores:
            raise ValueError("All models failed in ensemble")
        
        # Combine scores and labels
        ensemble_scores = np.mean(all_scores, axis=0)
        ensemble_labels = (np.mean(all_labels, axis=0) > 0.5).astype(int)  # Majority vote
        
        return AnomalyResult(
            timestamps=np.arange(len(X)),
            anomaly_scores=ensemble_scores,
            anomaly_labels=ensemble_labels,
            anomaly_types=[AnomalyType.POINT] * len(X),
            severity_levels=[self._score_to_severity(score) for score in ensemble_scores],
            confidence_scores=ensemble_scores,
            method_name='ensemble',
            threshold=0.5,
            metadata={'n_models': len(all_scores)}
        )
    
    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert anomaly score to severity level."""
        if score < 0.3:
            return AnomalySeverity.LOW
        elif score < 0.6:
            return AnomalySeverity.MEDIUM
        elif score < 0.8:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


class FusionAnomalyDetector:
    """
    Fusion-specific anomaly detection system.
    
    Combines statistical and ML methods with domain knowledge
    for comprehensive anomaly detection in fusion systems.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fusion anomaly detector.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config or {}
        self.statistical_detector = StatisticalAnomalyDetector(self.config.get('statistical', {}))
        self.ml_detector = MLAnomalyDetector(self.config.get('ml', {}))
        
        # Fusion-specific patterns
        self.fusion_patterns = self._initialize_fusion_patterns()
        
        # Detection history for temporal analysis
        self.detection_history = []
        
        logger.info("FusionAnomalyDetector initialized")
    
    def _initialize_fusion_patterns(self) -> List[FusionAnomalyPattern]:
        """Initialize fusion-specific anomaly patterns."""
        patterns = [
            FusionAnomalyPattern(
                name="disruption_precursor",
                description="Pattern indicating potential plasma disruption",
                parameters=["q_factor", "beta", "plasma_current"],
                detection_rules={
                    "q_factor": {"min": 0, "max": 2},
                    "beta": {"max": 0.1},
                    "plasma_current": {"rapid_decrease": True}
                },
                severity=AnomalySeverity.CRITICAL,
                response_actions=["emergency_shutdown", "magnetic_control"]
            ),
            FusionAnomalyPattern(
                name="confinement_degradation",
                description="Degradation in plasma confinement",
                parameters=["confinement_time", "energy_content"],
                detection_rules={
                    "confinement_time": {"decrease_rate": 0.1},
                    "energy_content": {"decrease_rate": 0.05}
                },
                severity=AnomalySeverity.HIGH,
                response_actions=["adjust_heating", "optimize_profile"]
            ),
            FusionAnomalyPattern(
                name="thermal_runaway",
                description="Uncontrolled temperature increase",
                parameters=["ion_temperature", "electron_temperature"],
                detection_rules={
                    "ion_temperature": {"increase_rate": 10.0},
                    "electron_temperature": {"increase_rate": 10.0}
                },
                severity=AnomalySeverity.CRITICAL,
                response_actions=["reduce_heating", "emergency_cooling"]
            ),
            FusionAnomalyPattern(
                name="density_limit",
                description="Approaching density limit",
                parameters=["electron_density", "line_density"],
                detection_rules={
                    "electron_density": {"threshold": 5e20},
                    "line_density": {"threshold": 1e20}
                },
                severity=AnomalySeverity.HIGH,
                response_actions=["gas_puffing_control", "pellet_injection"]
            )
        ]
        
        return patterns
    
    def detect_comprehensive(self, 
                           data: pd.DataFrame,
                           timestamp_col: str = 'timestamp') -> Dict[str, AnomalyResult]:
        """
        Comprehensive anomaly detection using multiple methods.
        
        Args:
            data: Input dataframe with fusion parameters.
            timestamp_col: Name of timestamp column.
            
        Returns:
            Dictionary of anomaly results from different methods.
        """
        results = {}
        
        # Prepare data
        if timestamp_col in data.columns:
            timestamps = data[timestamp_col]
            feature_data = data.drop(columns=[timestamp_col])
        else:
            timestamps = np.arange(len(data))
            feature_data = data
        
        feature_names = feature_data.columns.tolist()
        X = feature_data.values
        
        # Statistical methods for each feature
        for i, feature_name in enumerate(feature_names):
            feature_data_col = X[:, i]
            
            try:
                # Z-score detection
                z_result = self.statistical_detector.z_score_detection(feature_data_col)
                z_result.timestamps = timestamps
                results[f"{feature_name}_z_score"] = z_result
                
                # IQR detection
                iqr_result = self.statistical_detector.iqr_detection(feature_data_col)
                iqr_result.timestamps = timestamps
                results[f"{feature_name}_iqr"] = iqr_result
                
            except Exception as e:
                logger.warning(f"Statistical detection failed for {feature_name}: {e}")
        
        # ML-based multivariate detection
        if len(feature_names) > 1:
            try:
                # Fit and detect with Isolation Forest
                model_id = self.ml_detector.fit_isolation_forest(X)
                ml_result = self.ml_detector.detect_anomalies(X, model_id)
                ml_result.timestamps = timestamps
                results['isolation_forest'] = ml_result
                
                # Ensemble detection
                ensemble_id = self.ml_detector.fit_ensemble_detector(X)
                ensemble_result = self.ml_detector.detect_anomalies(X, ensemble_id)
                ensemble_result.timestamps = timestamps
                results['ensemble'] = ensemble_result
                
            except Exception as e:
                logger.warning(f"ML detection failed: {e}")
        
        # Fusion-specific pattern detection
        try:
            pattern_result = self.detect_fusion_patterns(feature_data)
            pattern_result.timestamps = timestamps
            results['fusion_patterns'] = pattern_result
            
        except Exception as e:
            logger.warning(f"Fusion pattern detection failed: {e}")
        
        # Store results in history
        self.detection_history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        logger.info(f"Comprehensive anomaly detection completed with {len(results)} methods")
        return results
    
    def detect_fusion_patterns(self, data: pd.DataFrame) -> AnomalyResult:
        """Detect fusion-specific anomaly patterns."""
        anomaly_scores = np.zeros(len(data))
        anomaly_labels = np.zeros(len(data))
        anomaly_types = [AnomalyType.CONTEXTUAL] * len(data)
        severity_levels = [AnomalySeverity.LOW] * len(data)
        
        detected_patterns = []
        
        for pattern in self.fusion_patterns:
            pattern_detected = self._check_pattern(data, pattern)
            
            if pattern_detected['detected']:
                detected_patterns.append(pattern.name)
                
                # Update anomaly scores for detected pattern
                for idx in pattern_detected['indices']:
                    anomaly_scores[idx] = max(anomaly_scores[idx], pattern_detected['confidence'])
                    anomaly_labels[idx] = 1
                    severity_levels[idx] = pattern.severity
                    
                    if pattern.name == "disruption_precursor":
                        anomaly_types[idx] = AnomalyType.CRITICAL
        
        return AnomalyResult(
            timestamps=np.arange(len(data)),
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_labels,
            anomaly_types=anomaly_types,
            severity_levels=severity_levels,
            confidence_scores=anomaly_scores,
            method_name='fusion_patterns',
            threshold=0.5,
            metadata={'detected_patterns': detected_patterns}
        )
    
    def _check_pattern(self, data: pd.DataFrame, pattern: FusionAnomalyPattern) -> Dict[str, Any]:
        """Check for specific fusion anomaly pattern."""
        detected_indices = []
        max_confidence = 0.0
        
        # Check if pattern parameters exist in data
        available_params = [p for p in pattern.parameters if p in data.columns]
        
        if not available_params:
            return {'detected': False, 'indices': [], 'confidence': 0.0}
        
        # Apply detection rules
        for param in available_params:
            param_data = data[param].values
            rules = pattern.detection_rules.get(param, {})
            
            # Check thresholds
            if 'min' in rules:
                violations = param_data < rules['min']
                detected_indices.extend(np.where(violations)[0])
                
            if 'max' in rules:
                violations = param_data > rules['max']
                detected_indices.extend(np.where(violations)[0])
            
            # Check rate of change
            if 'increase_rate' in rules or 'decrease_rate' in rules:
                if len(param_data) > 1:
                    rates = np.diff(param_data)
                    
                    if 'increase_rate' in rules:
                        violations = rates > rules['increase_rate']
                        detected_indices.extend(np.where(violations)[0] + 1)
                    
                    if 'decrease_rate' in rules:
                        violations = rates < -rules['decrease_rate']
                        detected_indices.extend(np.where(violations)[0] + 1)
        
        # Remove duplicates and calculate confidence
        detected_indices = list(set(detected_indices))
        
        if detected_indices:
            confidence = min(1.0, len(detected_indices) / max(10, len(data) * 0.1))
            max_confidence = confidence
        
        return {
            'detected': len(detected_indices) > 0,
            'indices': detected_indices,
            'confidence': max_confidence
        }
    
    def get_anomaly_summary(self, results: Dict[str, AnomalyResult]) -> Dict[str, Any]:
        """Generate summary of anomaly detection results."""
        summary = {
            'total_methods': len(results),
            'anomaly_counts': {},
            'severity_distribution': {},
            'consensus_anomalies': [],
            'critical_periods': []
        }
        
        # Count anomalies by method
        for method, result in results.items():
            anomaly_count = np.sum(result.anomaly_labels)
            summary['anomaly_counts'][method] = anomaly_count
        
        # Severity distribution
        all_severities = []
        for result in results.values():
            all_severities.extend([s.value for s in result.severity_levels])
        
        for severity in AnomalySeverity:
            summary['severity_distribution'][severity.value] = all_severities.count(severity.value)
        
        # Find consensus anomalies (detected by multiple methods)
        if len(results) > 1:
            # Find time points where multiple methods agree
            all_labels = np.array([result.anomaly_labels for result in results.values()])
            consensus_threshold = max(2, len(results) // 2)
            consensus_anomalies = np.sum(all_labels, axis=0) >= consensus_threshold
            summary['consensus_anomalies'] = np.where(consensus_anomalies)[0].tolist()
        
        return summary
    
    def save_models(self, filepath: str):
        """Save trained models."""
        try:
            save_data = {
                'ml_models': self.ml_detector.models,
                'ml_scalers': self.ml_detector.scalers,
                'config': self.config
            }
            joblib.dump(save_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        try:
            save_data = joblib.load(filepath)
            self.ml_detector.models = save_data['ml_models']
            self.ml_detector.scalers = save_data['ml_scalers']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


def create_fusion_anomaly_detector(config_path: Optional[str] = None) -> FusionAnomalyDetector:
    """
    Create fusion anomaly detector with configuration.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configured FusionAnomalyDetector.
    """
    config = {}
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config().get('anomaly_detection', {})
    
    return FusionAnomalyDetector(config)