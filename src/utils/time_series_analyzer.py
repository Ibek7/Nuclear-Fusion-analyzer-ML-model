"""
Advanced time series analysis for nuclear fusion data.

This module provides comprehensive time series analysis capabilities
for fusion reactor data including forecasting, pattern detection,
changepoint analysis, and temporal anomaly detection.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

try:
    # Statistical and time series libraries
    from scipy import stats, signal
    from scipy.fft import fft, fftfreq
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # Advanced time series analysis
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    # Change point detection
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    # Prophet for forecasting
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    # Advanced ML for time series
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesFeatures:
    """Container for time series features."""
    
    # Statistical features
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    
    # Trend features
    trend_slope: float
    trend_r2: float
    
    # Seasonality features
    seasonal_strength: float
    period: Optional[int]
    
    # Frequency domain features
    dominant_frequency: float
    spectral_entropy: float
    
    # Stationarity tests
    adf_statistic: float
    adf_p_value: float
    kpss_statistic: float
    kpss_p_value: float
    is_stationary: bool
    
    # Change points
    change_points: List[int]
    n_change_points: int
    
    # Anomaly information
    anomaly_score: float
    anomaly_periods: List[Tuple[int, int]]


@dataclass
class ForecastResult:
    """Container for forecasting results."""
    
    timestamps: np.ndarray
    forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    model_name: str
    metrics: Dict[str, float]


class TimeSeriesProcessor:
    """
    Advanced time series processor for fusion reactor data.
    
    Provides preprocessing, feature extraction, and analysis
    capabilities for temporal fusion parameters.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the time series processor.
        
        Args:
            config: Configuration dictionary for processing parameters.
        """
        self.config = config or {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Processing parameters
        self.sampling_rate = self.config.get('sampling_rate', 1.0)  # Hz
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.filter_type = self.config.get('filter_type', 'butter')
        self.filter_order = self.config.get('filter_order', 5)
        
        logger.info("TimeSeriesProcessor initialized")
    
    def preprocess(self, 
                   data: pd.DataFrame, 
                   timestamp_col: str = 'timestamp',
                   fill_method: str = 'interpolate') -> pd.DataFrame:
        """
        Preprocess time series data.
        
        Args:
            data: Input dataframe with time series data.
            timestamp_col: Name of timestamp column.
            fill_method: Method for handling missing values.
            
        Returns:
            Preprocessed dataframe.
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Handle missing values
        if fill_method == 'interpolate':
            df = df.interpolate(method=self.interpolation_method)
        elif fill_method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif fill_method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif fill_method == 'drop':
            df = df.dropna()
        
        # Remove duplicated timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Resample if needed
        if 'resample_freq' in self.config:
            df = df.resample(self.config['resample_freq']).mean()
        
        logger.info(f"Preprocessed time series data: {df.shape}")
        return df
    
    def apply_filter(self, 
                     series: pd.Series, 
                     filter_type: str = 'lowpass',
                     cutoff_freq: float = 0.1) -> pd.Series:
        """
        Apply digital filter to time series.
        
        Args:
            series: Input time series.
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass').
            cutoff_freq: Cutoff frequency (normalized to Nyquist).
            
        Returns:
            Filtered time series.
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available, returning original series")
            return series
        
        try:
            # Design filter
            if filter_type == 'lowpass':
                b, a = signal.butter(self.filter_order, cutoff_freq, btype='low')
            elif filter_type == 'highpass':
                b, a = signal.butter(self.filter_order, cutoff_freq, btype='high')
            elif filter_type == 'bandpass':
                # cutoff_freq should be a tuple for bandpass
                if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                    b, a = signal.butter(self.filter_order, cutoff_freq, btype='band')
                else:
                    raise ValueError("Bandpass filter requires two cutoff frequencies")
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            # Apply filter
            filtered_data = signal.filtfilt(b, a, series.values)
            
            return pd.Series(filtered_data, index=series.index, name=series.name)
            
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return series
    
    def extract_features(self, series: pd.Series) -> TimeSeriesFeatures:
        """
        Extract comprehensive time series features.
        
        Args:
            series: Input time series.
            
        Returns:
            TimeSeriesFeatures object.
        """
        # Remove NaN values for feature calculation
        clean_series = series.dropna()
        values = clean_series.values
        
        if len(values) < 10:
            logger.warning("Insufficient data for feature extraction")
            # Return default features
            return TimeSeriesFeatures(
                mean=0.0, std=0.0, variance=0.0, skewness=0.0, kurtosis=0.0,
                trend_slope=0.0, trend_r2=0.0, seasonal_strength=0.0, period=None,
                dominant_frequency=0.0, spectral_entropy=0.0,
                adf_statistic=0.0, adf_p_value=1.0, kpss_statistic=0.0, kpss_p_value=1.0,
                is_stationary=False, change_points=[], n_change_points=0,
                anomaly_score=0.0, anomaly_periods=[]
            )
        
        # Statistical features
        mean_val = np.mean(values)
        std_val = np.std(values)
        variance_val = np.var(values)
        skewness_val = stats.skew(values)
        kurtosis_val = stats.kurtosis(values)
        
        # Trend analysis
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        trend_slope = slope
        trend_r2 = r_value ** 2
        
        # Frequency domain analysis
        if HAS_SCIPY and len(values) > 1:
            fft_vals = fft(values)
            freqs = fftfreq(len(values), d=1/self.sampling_rate)
            power_spectrum = np.abs(fft_vals) ** 2
            
            # Dominant frequency
            positive_freqs = freqs[freqs > 0]
            positive_power = power_spectrum[freqs > 0]
            if len(positive_power) > 0:
                dominant_freq_idx = np.argmax(positive_power)
                dominant_frequency = positive_freqs[dominant_freq_idx]
            else:
                dominant_frequency = 0.0
            
            # Spectral entropy
            normalized_power = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-12))
        else:
            dominant_frequency = 0.0
            spectral_entropy = 0.0
        
        # Seasonality detection
        seasonal_strength = 0.0
        period = None
        
        if HAS_STATSMODELS and len(values) >= 24:  # Need sufficient data for seasonal decomposition
            try:
                # Try multiple potential periods
                potential_periods = [12, 24, 48, 96]  # Common periods for different time scales
                max_strength = 0.0
                best_period = None
                
                for p in potential_periods:
                    if len(values) >= 2 * p:
                        try:
                            decomposition = seasonal_decompose(
                                clean_series, 
                                model='additive', 
                                period=p,
                                extrapolate_trend='freq'
                            )
                            # Calculate seasonal strength
                            seasonal_var = np.var(decomposition.seasonal.dropna())
                            remainder_var = np.var(decomposition.resid.dropna())
                            if remainder_var > 0:
                                strength = seasonal_var / (seasonal_var + remainder_var)
                                if strength > max_strength:
                                    max_strength = strength
                                    best_period = p
                        except:
                            continue
                
                seasonal_strength = max_strength
                period = best_period
                
            except Exception as e:
                logger.debug(f"Seasonal decomposition failed: {e}")
        
        # Stationarity tests
        adf_statistic = 0.0
        adf_p_value = 1.0
        kpss_statistic = 0.0
        kpss_p_value = 1.0
        is_stationary = False
        
        if HAS_STATSMODELS:
            try:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(values)
                adf_statistic = adf_result[0]
                adf_p_value = adf_result[1]
                
                # KPSS test
                kpss_result = kpss(values)
                kpss_statistic = kpss_result[0]
                kpss_p_value = kpss_result[1]
                
                # Series is stationary if ADF rejects null (p < 0.05) and KPSS fails to reject null (p > 0.05)
                is_stationary = (adf_p_value < 0.05) and (kpss_p_value > 0.05)
                
            except Exception as e:
                logger.debug(f"Stationarity tests failed: {e}")
        
        # Change point detection
        change_points = []
        if HAS_RUPTURES and len(values) > 10:
            try:
                algo = rpt.Pelt(model="rbf").fit(values)
                change_points = algo.predict(pen=10)
                # Remove the last point (end of series)
                if change_points and change_points[-1] == len(values):
                    change_points = change_points[:-1]
            except Exception as e:
                logger.debug(f"Change point detection failed: {e}")
        
        # Anomaly detection
        anomaly_score = 0.0
        anomaly_periods = []
        
        if HAS_SKLEARN and len(values) > 10:
            try:
                # Use Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = iso_forest.decision_function(values.reshape(-1, 1))
                anomaly_score = np.mean(np.abs(anomaly_scores))
                
                # Find anomalous periods
                is_anomaly = iso_forest.predict(values.reshape(-1, 1)) == -1
                anomaly_indices = np.where(is_anomaly)[0]
                
                # Group consecutive anomalies into periods
                if len(anomaly_indices) > 0:
                    periods = []
                    start = anomaly_indices[0]
                    end = start
                    
                    for i in range(1, len(anomaly_indices)):
                        if anomaly_indices[i] == anomaly_indices[i-1] + 1:
                            end = anomaly_indices[i]
                        else:
                            periods.append((start, end))
                            start = anomaly_indices[i]
                            end = start
                    
                    periods.append((start, end))
                    anomaly_periods = periods
                    
            except Exception as e:
                logger.debug(f"Anomaly detection failed: {e}")
        
        return TimeSeriesFeatures(
            mean=mean_val,
            std=std_val,
            variance=variance_val,
            skewness=skewness_val,
            kurtosis=kurtosis_val,
            trend_slope=trend_slope,
            trend_r2=trend_r2,
            seasonal_strength=seasonal_strength,
            period=period,
            dominant_frequency=dominant_frequency,
            spectral_entropy=spectral_entropy,
            adf_statistic=adf_statistic,
            adf_p_value=adf_p_value,
            kpss_statistic=kpss_statistic,
            kpss_p_value=kpss_p_value,
            is_stationary=is_stationary,
            change_points=change_points,
            n_change_points=len(change_points),
            anomaly_score=anomaly_score,
            anomaly_periods=anomaly_periods
        )


class FusionTimeSeriesForecaster:
    """
    Advanced forecasting system for fusion reactor parameters.
    
    Supports multiple forecasting models including ARIMA, SARIMA,
    Prophet, and ensemble methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the forecaster.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.models = {}
        self.processor = TimeSeriesProcessor(config)
        
        # Forecasting parameters
        self.forecast_horizon = self.config.get('forecast_horizon', 24)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        logger.info("FusionTimeSeriesForecaster initialized")
    
    def fit_arima(self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Fit ARIMA model to time series.
        
        Args:
            series: Input time series.
            order: ARIMA order (p, d, q). If None, will be auto-determined.
            
        Returns:
            Fitted model information.
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for ARIMA modeling")
        
        clean_series = series.dropna()
        
        try:
            if order is None:
                # Auto ARIMA - simple grid search
                best_aic = np.inf
                best_order = (1, 1, 1)
                
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                model = ARIMA(clean_series, order=(p, d, q))
                                fitted_model = model.fit()
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                order = best_order
            
            # Fit final model
            model = ARIMA(clean_series, order=order)
            fitted_model = model.fit()
            
            self.models['arima'] = fitted_model
            
            return {
                'model': fitted_model,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': fitted_model.params.to_dict()
            }
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise
    
    def fit_prophet(self, series: pd.Series) -> Dict:
        """
        Fit Prophet model to time series.
        
        Args:
            series: Input time series.
            
        Returns:
            Fitted model information.
        """
        if not HAS_PROPHET:
            raise ImportError("prophet is required for Prophet modeling")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        try:
            # Create and fit model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)
            
            self.models['prophet'] = model
            
            return {
                'model': model,
                'params': model.params
            }
            
        except Exception as e:
            logger.error(f"Prophet fitting failed: {e}")
            raise
    
    def forecast(self, 
                 model_name: str, 
                 steps: Optional[int] = None,
                 series: Optional[pd.Series] = None) -> ForecastResult:
        """
        Generate forecasts using specified model.
        
        Args:
            model_name: Name of the model to use.
            steps: Number of steps to forecast.
            series: Original series for timestamp generation.
            
        Returns:
            ForecastResult object.
        """
        steps = steps or self.forecast_horizon
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not fitted")
        
        model = self.models[model_name]
        
        try:
            if model_name == 'arima':
                forecast_result = model.forecast(steps=steps)
                conf_int = model.get_forecast(steps=steps).conf_int(alpha=1-self.confidence_level)
                
                # Generate timestamps
                if series is not None and len(series) > 0:
                    last_timestamp = series.index[-1]
                    freq = pd.infer_freq(series.index)
                    if freq is None:
                        # Estimate frequency from mean time difference
                        time_diffs = np.diff(series.index)
                        mean_diff = np.mean(time_diffs)
                        timestamps = pd.date_range(
                            start=last_timestamp + mean_diff,
                            periods=steps,
                            freq=mean_diff
                        )
                    else:
                        timestamps = pd.date_range(
                            start=last_timestamp,
                            periods=steps+1,
                            freq=freq
                        )[1:]
                else:
                    timestamps = np.arange(steps)
                
                return ForecastResult(
                    timestamps=timestamps,
                    forecast=forecast_result.values,
                    lower_bound=conf_int.iloc[:, 0].values,
                    upper_bound=conf_int.iloc[:, 1].values,
                    confidence_level=self.confidence_level,
                    model_name='ARIMA',
                    metrics={'aic': model.aic, 'bic': model.bic}
                )
            
            elif model_name == 'prophet':
                # Create future dataframe
                future = model.make_future_dataframe(periods=steps)
                forecast = model.predict(future)
                
                # Extract forecast portion
                forecast_data = forecast.tail(steps)
                
                return ForecastResult(
                    timestamps=forecast_data['ds'].values,
                    forecast=forecast_data['yhat'].values,
                    lower_bound=forecast_data['yhat_lower'].values,
                    upper_bound=forecast_data['yhat_upper'].values,
                    confidence_level=self.confidence_level,
                    model_name='Prophet',
                    metrics={}
                )
            
            else:
                raise ValueError(f"Forecasting not implemented for model: {model_name}")
                
        except Exception as e:
            logger.error(f"Forecasting failed for model {model_name}: {e}")
            raise
    
    def ensemble_forecast(self, 
                          series: pd.Series, 
                          models: Optional[List[str]] = None) -> ForecastResult:
        """
        Generate ensemble forecast from multiple models.
        
        Args:
            series: Input time series.
            models: List of model names to ensemble.
            
        Returns:
            Ensemble ForecastResult.
        """
        models = models or ['arima', 'prophet']
        available_models = [m for m in models if m in self.models]
        
        if not available_models:
            raise ValueError("No available models for ensemble")
        
        forecasts = []
        for model_name in available_models:
            try:
                result = self.forecast(model_name, series=series)
                forecasts.append(result)
            except Exception as e:
                logger.warning(f"Model {model_name} failed in ensemble: {e}")
        
        if not forecasts:
            raise ValueError("All models failed in ensemble")
        
        # Simple average ensemble
        ensemble_forecast = np.mean([f.forecast for f in forecasts], axis=0)
        ensemble_lower = np.mean([f.lower_bound for f in forecasts], axis=0)
        ensemble_upper = np.mean([f.upper_bound for f in forecasts], axis=0)
        
        # Use timestamps from first successful forecast
        timestamps = forecasts[0].timestamps
        
        return ForecastResult(
            timestamps=timestamps,
            forecast=ensemble_forecast,
            lower_bound=ensemble_lower,
            upper_bound=ensemble_upper,
            confidence_level=self.confidence_level,
            model_name='Ensemble',
            metrics={'n_models': len(forecasts)}
        )
    
    def save_models(self, filepath: str):
        """Save all fitted models to file."""
        try:
            joblib.dump(self.models, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self, filepath: str):
        """Load models from file."""
        try:
            self.models = joblib.load(filepath)
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


def create_time_series_analyzer(config_path: Optional[str] = None) -> Tuple[TimeSeriesProcessor, FusionTimeSeriesForecaster]:
    """
    Create time series analysis components.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Tuple of (processor, forecaster).
    """
    config = {}
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config().get('time_series', {})
    
    processor = TimeSeriesProcessor(config)
    forecaster = FusionTimeSeriesForecaster(config)
    
    return processor, forecaster