"""
Business Intelligence Module for Nuclear Fusion Analysis.

This module provides:
- Executive dashboards and reporting
- Data warehousing and OLAP cubes
- Advanced analytics and data mining
- Predictive insights and forecasting
- Business metrics and ROI analysis
- Strategic planning support
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

import numpy as np
import pandas as pd

# Advanced analytics
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of business intelligence analysis."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class DataGranularity(Enum):
    """Data aggregation granularity levels."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class BusinessMetric:
    """Business metric definition."""
    
    name: str
    formula: str
    description: str
    category: str
    target_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    improvement_direction: str = "higher"  # "higher", "lower", or "stable"
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """Calculate metric value from data."""
        try:
            # Simple formula evaluation (extend as needed)
            if self.formula in data:
                return float(data[self.formula])
            
            # Support basic arithmetic operations
            formula = self.formula
            for key, value in data.items():
                formula = formula.replace(key, str(value))
            
            return eval(formula)
        except Exception as e:
            logger.error(f"Error calculating metric {self.name}: {e}")
            return 0.0


@dataclass
class DataCube:
    """OLAP data cube for multidimensional analysis."""
    
    name: str
    dimensions: List[str]
    measures: List[str]
    data: pd.DataFrame
    aggregations: Dict[str, str] = field(default_factory=dict)  # measure -> aggregation function
    
    def __post_init__(self):
        """Initialize aggregations if not provided."""
        if not self.aggregations:
            for measure in self.measures:
                self.aggregations[measure] = "sum"
    
    def slice(self, dimension: str, value: Any) -> "DataCube":
        """Create a slice of the cube by fixing one dimension."""
        filtered_data = self.data[self.data[dimension] == value].copy()
        remaining_dimensions = [d for d in self.dimensions if d != dimension]
        
        return DataCube(
            name=f"{self.name}_slice_{dimension}_{value}",
            dimensions=remaining_dimensions,
            measures=self.measures,
            data=filtered_data,
            aggregations=self.aggregations
        )
    
    def dice(self, filters: Dict[str, Any]) -> "DataCube":
        """Create a dice of the cube by applying multiple filters."""
        filtered_data = self.data.copy()
        
        for dimension, value in filters.items():
            if dimension in self.dimensions:
                if isinstance(value, list):
                    filtered_data = filtered_data[filtered_data[dimension].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[dimension] == value]
        
        return DataCube(
            name=f"{self.name}_dice",
            dimensions=self.dimensions,
            measures=self.measures,
            data=filtered_data,
            aggregations=self.aggregations
        )
    
    def drill_down(self, dimension: str, granularity: DataGranularity) -> "DataCube":
        """Drill down to more detailed level."""
        # Implementation depends on data structure
        # This is a simplified version
        return self
    
    def roll_up(self, dimension: str) -> "DataCube":
        """Roll up to higher aggregation level."""
        # Group by remaining dimensions and aggregate measures
        remaining_dims = [d for d in self.dimensions if d != dimension]
        
        if not remaining_dims:
            # Aggregate all data
            aggregated = {}
            for measure in self.measures:
                agg_func = self.aggregations.get(measure, "sum")
                if agg_func == "sum":
                    aggregated[measure] = [self.data[measure].sum()]
                elif agg_func == "mean":
                    aggregated[measure] = [self.data[measure].mean()]
                elif agg_func == "count":
                    aggregated[measure] = [self.data[measure].count()]
            
            aggregated_data = pd.DataFrame(aggregated)
        else:
            # Group by remaining dimensions
            agg_dict = {}
            for measure in self.measures:
                agg_func = self.aggregations.get(measure, "sum")
                agg_dict[measure] = agg_func
            
            aggregated_data = self.data.groupby(remaining_dims).agg(agg_dict).reset_index()
        
        return DataCube(
            name=f"{self.name}_rollup",
            dimensions=remaining_dims,
            measures=self.measures,
            data=aggregated_data,
            aggregations=self.aggregations
        )
    
    def pivot(self, rows: List[str], columns: List[str], values: str) -> pd.DataFrame:
        """Create pivot table from cube data."""
        return pd.pivot_table(
            self.data,
            index=rows,
            columns=columns,
            values=values,
            aggfunc=self.aggregations.get(values, "sum"),
            fill_value=0
        )


class DataMiner:
    """Advanced data mining and pattern discovery."""
    
    def __init__(self):
        """Initialize data miner."""
        self.models = {}
        self.scalers = {}
        
        logger.info("DataMiner initialized")
    
    def find_clusters(
        self,
        data: pd.DataFrame,
        features: List[str],
        method: str = "kmeans",
        n_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Find clusters in data.
        
        Args:
            data: Input data.
            features: Features to use for clustering.
            method: Clustering method (kmeans, dbscan).
            n_clusters: Number of clusters (for kmeans).
            
        Returns:
            Cluster labels and metadata.
        """
        if not HAS_ML:
            raise RuntimeError("Scikit-learn not available")
        
        # Prepare data
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == "kmeans":
            if n_clusters is None:
                # Find optimal number of clusters using elbow method
                inertias = []
                silhouette_scores = []
                k_range = range(2, min(11, len(data) // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    inertias.append(kmeans.inertia_)
                    
                    if len(set(cluster_labels)) > 1:
                        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
                    else:
                        silhouette_scores.append(0)
                
                # Choose k with best silhouette score
                n_clusters = k_range[np.argmax(silhouette_scores)]
            
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            
            metadata = {
                "method": method,
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
                "centers": model.cluster_centers_.tolist(),
                "silhouette_score": silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            }
        
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            metadata = {
                "method": method,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
            }
        
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        self.models[f"cluster_{method}"] = model
        self.scalers[f"cluster_{method}"] = scaler
        
        return labels, metadata
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        features: List[str],
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in data.
        
        Args:
            data: Input data.
            features: Features to use for anomaly detection.
            contamination: Expected proportion of anomalies.
            
        Returns:
            Anomaly labels (-1 for anomalies, 1 for normal) and metadata.
        """
        if not HAS_ML:
            raise RuntimeError("Scikit-learn not available")
        
        # Prepare data
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(X_scaled)
        
        n_anomalies = (labels == -1).sum()
        
        metadata = {
            "method": "isolation_forest",
            "contamination": contamination,
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(n_anomalies / len(data))
        }
        
        self.models["anomaly_detection"] = model
        self.scalers["anomaly_detection"] = scaler
        
        return labels, metadata
    
    def find_patterns(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str],
        method: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Find patterns and feature importance.
        
        Args:
            data: Input data.
            target: Target variable.
            features: Feature variables.
            method: Method to use (random_forest, xgboost).
            
        Returns:
            Pattern analysis results.
        """
        if not HAS_ML:
            raise RuntimeError("Scikit-learn not available")
        
        X = data[features].values
        y = data[target].values
        
        if method == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = dict(zip(features, model.feature_importances_))
            
            results = {
                "method": method,
                "feature_importance": feature_importance,
                "score": model.score(X, y),
                "top_features": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        elif method == "xgboost" and HAS_ML:
            try:
                model = xgb.XGBRegressor(random_state=42)
                model.fit(X, y)
                
                feature_importance = dict(zip(features, model.feature_importances_))
                
                results = {
                    "method": method,
                    "feature_importance": feature_importance,
                    "score": model.score(X, y),
                    "top_features": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                }
            except ImportError:
                raise RuntimeError("XGBoost not available")
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.models[f"pattern_{method}"] = model
        
        return results


class ForecastEngine:
    """Time series forecasting and prediction engine."""
    
    def __init__(self):
        """Initialize forecast engine."""
        self.models = {}
        
        logger.info("ForecastEngine initialized")
    
    def decompose_series(self, data: pd.Series, period: int = None) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data: Time series data.
            period: Seasonal period.
            
        Returns:
            Decomposition results.
        """
        if not HAS_STATSMODELS:
            raise RuntimeError("Statsmodels not available")
        
        if period is None:
            # Try to detect period automatically
            period = min(12, len(data) // 2)
        
        decomposition = seasonal_decompose(data, model='additive', period=period)
        
        return {
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "residual": decomposition.resid.dropna().tolist(),
            "period": period
        }
    
    def forecast_arima(
        self,
        data: pd.Series,
        steps: int = 10,
        order: Tuple[int, int, int] = (1, 1, 1)
    ) -> Dict[str, Any]:
        """
        Generate ARIMA forecast.
        
        Args:
            data: Time series data.
            steps: Number of steps to forecast.
            order: ARIMA order (p, d, q).
            
        Returns:
            Forecast results.
        """
        if not HAS_STATSMODELS:
            raise RuntimeError("Statsmodels not available")
        
        try:
            model = ARIMA(data, order=order)
            fitted = model.fit()
            
            forecast = fitted.forecast(steps=steps)
            conf_int = fitted.get_forecast(steps=steps).conf_int()
            
            results = {
                "method": "arima",
                "order": order,
                "forecast": forecast.tolist(),
                "confidence_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "aic": fitted.aic,
                "bic": fitted.bic
            }
            
            self.models["arima"] = fitted
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return {"error": str(e)}
    
    def forecast_ml(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str],
        steps: int = 10,
        method: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Generate ML-based forecast.
        
        Args:
            data: Input data with features.
            target: Target variable to forecast.
            features: Feature variables.
            steps: Number of steps to forecast.
            method: ML method to use.
            
        Returns:
            Forecast results.
        """
        if not HAS_ML:
            raise RuntimeError("Scikit-learn not available")
        
        X = data[features].values
        y = data[target].values
        
        if method == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Generate future predictions (simplified approach)
            last_features = X[-1:].copy()
            forecasts = []
            
            for _ in range(steps):
                pred = model.predict(last_features)[0]
                forecasts.append(pred)
                
                # Update features for next prediction (simplified)
                # In practice, this would be more sophisticated
                last_features = np.roll(last_features, -1)
                last_features[0, -1] = pred
            
            results = {
                "method": method,
                "forecast": forecasts,
                "score": model.score(X, y)
            }
            
            self.models[f"ml_{method}"] = model
            
            return results
        
        else:
            raise ValueError(f"Unsupported method: {method}")


class BusinessIntelligence:
    """
    Comprehensive business intelligence system.
    
    Provides advanced analytics, data mining, and business insights.
    """
    
    def __init__(self):
        """Initialize business intelligence system."""
        self.data_cubes: Dict[str, DataCube] = {}
        self.business_metrics: Dict[str, BusinessMetric] = {}
        self.data_miner = DataMiner()
        self.forecast_engine = ForecastEngine()
        
        # Setup default business metrics
        self._setup_default_metrics()
        
        logger.info("BusinessIntelligence initialized")
    
    def _setup_default_metrics(self):
        """Setup default business metrics for fusion analysis."""
        default_metrics = [
            BusinessMetric(
                name="Energy Efficiency",
                formula="energy_output / energy_input",
                description="Ratio of energy output to input",
                category="Performance",
                target_value=1.0,
                improvement_direction="higher"
            ),
            BusinessMetric(
                name="Operational Cost per Hour",
                formula="total_operational_cost / operating_hours",
                description="Average operational cost per hour",
                category="Financial",
                improvement_direction="lower"
            ),
            BusinessMetric(
                name="Prediction Accuracy Rate",
                formula="correct_predictions / total_predictions",
                description="Rate of correct ML predictions",
                category="Quality",
                target_value=0.95,
                improvement_direction="higher"
            ),
            BusinessMetric(
                name="System Availability",
                formula="uptime_hours / total_hours",
                description="System availability percentage",
                category="Reliability",
                target_value=0.999,
                improvement_direction="higher"
            ),
            BusinessMetric(
                name="Data Processing Throughput",
                formula="processed_data_points / processing_time",
                description="Data processing rate",
                category="Performance",
                improvement_direction="higher"
            )
        ]
        
        for metric in default_metrics:
            self.business_metrics[metric.name] = metric
    
    def add_data_cube(self, cube: DataCube):
        """Add data cube to BI system."""
        self.data_cubes[cube.name] = cube
        
        logger.info(f"Data cube '{cube.name}' added")
    
    def add_business_metric(self, metric: BusinessMetric):
        """Add business metric definition."""
        self.business_metrics[metric.name] = metric
        
        logger.info(f"Business metric '{metric.name}' added")
    
    def calculate_business_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all business metrics."""
        results = {}
        
        for name, metric in self.business_metrics.items():
            try:
                value = metric.calculate(data)
                results[name] = value
            except Exception as e:
                logger.error(f"Error calculating metric {name}: {e}")
                results[name] = 0.0
        
        return results
    
    def generate_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary with key insights."""
        # Calculate business metrics
        metrics = self.calculate_business_metrics(data)
        
        # Identify key performance indicators
        kpi_status = {}
        for name, value in metrics.items():
            metric_def = self.business_metrics[name]
            
            if metric_def.target_value:
                performance = value / metric_def.target_value
                
                if performance >= 1.0:
                    status = "excellent"
                elif performance >= 0.9:
                    status = "good"
                elif performance >= 0.8:
                    status = "warning"
                else:
                    status = "critical"
                
                kpi_status[name] = {
                    "value": value,
                    "target": metric_def.target_value,
                    "performance": performance,
                    "status": status
                }
        
        # Generate insights
        insights = []
        
        # Performance insights
        excellent_kpis = [k for k, v in kpi_status.items() if v["status"] == "excellent"]
        critical_kpis = [k for k, v in kpi_status.items() if v["status"] == "critical"]
        
        if excellent_kpis:
            insights.append(f"Excellent performance in: {', '.join(excellent_kpis)}")
        
        if critical_kpis:
            insights.append(f"Critical attention needed for: {', '.join(critical_kpis)}")
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "kpi_status": kpi_status,
            "insights": insights,
            "overall_score": sum(v["performance"] for v in kpi_status.values()) / len(kpi_status) if kpi_status else 0
        }
    
    def perform_cluster_analysis(
        self,
        cube_name: str,
        features: List[str],
        method: str = "kmeans"
    ) -> Dict[str, Any]:
        """Perform cluster analysis on data cube."""
        if cube_name not in self.data_cubes:
            raise ValueError(f"Data cube '{cube_name}' not found")
        
        cube = self.data_cubes[cube_name]
        labels, metadata = self.data_miner.find_clusters(cube.data, features, method)
        
        # Add cluster labels to cube data
        cube.data["cluster"] = labels
        
        # Generate cluster summary
        cluster_summary = {}
        for cluster_id in set(labels):
            cluster_data = cube.data[cube.data["cluster"] == cluster_id]
            cluster_summary[cluster_id] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(cube.data) * 100,
                "characteristics": {
                    feature: {
                        "mean": cluster_data[feature].mean(),
                        "std": cluster_data[feature].std()
                    }
                    for feature in features if feature in cluster_data.columns
                }
            }
        
        return {
            "method": method,
            "metadata": metadata,
            "cluster_summary": cluster_summary,
            "labels": labels.tolist()
        }
    
    def generate_forecast(
        self,
        cube_name: str,
        target: str,
        features: List[str] = None,
        steps: int = 10,
        method: str = "arima"
    ) -> Dict[str, Any]:
        """Generate forecast for business metrics."""
        if cube_name not in self.data_cubes:
            raise ValueError(f"Data cube '{cube_name}' not found")
        
        cube = self.data_cubes[cube_name]
        
        if method == "arima":
            if target not in cube.data.columns:
                raise ValueError(f"Target '{target}' not found in cube data")
            
            series = cube.data[target]
            return self.forecast_engine.forecast_arima(series, steps)
        
        elif method in ["random_forest", "xgboost"]:
            if not features:
                features = [col for col in cube.data.columns if col != target]
            
            return self.forecast_engine.forecast_ml(cube.data, target, features, steps, method)
        
        else:
            raise ValueError(f"Unsupported forecast method: {method}")


def create_business_intelligence() -> BusinessIntelligence:
    """
    Create configured business intelligence system.
    
    Returns:
        Configured BI system.
    """
    return BusinessIntelligence()