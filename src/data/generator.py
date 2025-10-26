"""
Nuclear fusion data generator for creating synthetic plasma and reactor data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FusionDataGenerator:
    """
    Generator for synthetic nuclear fusion data including plasma parameters,
    reactor conditions, and fusion efficiency metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the fusion data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.random_seed = self.config.get('random_seed', 42)
        self.noise_level = self.config.get('noise_level', 0.1)
        
        np.random.seed(self.random_seed)
        
        # Physical constants and ranges for fusion parameters
        self.parameter_ranges = {
            'plasma_temperature': (1e8, 5e8),  # Kelvin
            'plasma_density': (1e20, 1e21),    # particles/m^3
            'magnetic_field': (5, 15),          # Tesla
            'confinement_time': (0.1, 2.0),    # seconds
            'pressure': (1e5, 1e7),            # Pascal
            'beta_plasma': (0.01, 0.1),        # dimensionless
            'safety_factor': (1.5, 4.0),       # dimensionless
            'neutral_beam_power': (10, 100),   # MW
            'rf_heating_power': (5, 50),       # MW
            'fuel_injection_rate': (1e22, 1e24),  # particles/s
        }
    
    def generate_plasma_parameters(self, n_samples: int) -> pd.DataFrame:
        """
        Generate realistic plasma parameters with physical correlations.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with plasma parameters
        """
        data = {}
        
        # Generate primary parameters
        data['plasma_temperature'] = np.random.uniform(
            *self.parameter_ranges['plasma_temperature'], n_samples
        )
        
        data['plasma_density'] = np.random.uniform(
            *self.parameter_ranges['plasma_density'], n_samples
        )
        
        data['magnetic_field'] = np.random.uniform(
            *self.parameter_ranges['magnetic_field'], n_samples
        )
        
        # Generate correlated parameters
        # Pressure correlates with temperature and density
        data['pressure'] = (
            data['plasma_temperature'] * data['plasma_density'] * 1.38e-23 +
            np.random.normal(0, self.noise_level * 1e6, n_samples)
        )
        
        # Confinement time correlates with magnetic field and temperature
        data['confinement_time'] = (
            0.5 * (data['magnetic_field'] / 10) * 
            np.sqrt(data['plasma_temperature'] / 1e8) +
            np.random.normal(0, self.noise_level * 0.1, n_samples)
        )
        
        # Beta plasma (ratio of plasma pressure to magnetic pressure)
        data['beta_plasma'] = (
            data['pressure'] / (data['magnetic_field']**2 / (2 * 4e-7 * np.pi)) +
            np.random.normal(0, self.noise_level * 0.01, n_samples)
        )
        
        # Safety factor (related to magnetic field configuration)
        data['safety_factor'] = np.random.uniform(
            *self.parameter_ranges['safety_factor'], n_samples
        )
        
        return pd.DataFrame(data)
    
    def generate_heating_systems(self, n_samples: int) -> pd.DataFrame:
        """
        Generate heating system parameters.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with heating parameters
        """
        data = {}
        
        data['neutral_beam_power'] = np.random.uniform(
            *self.parameter_ranges['neutral_beam_power'], n_samples
        )
        
        data['rf_heating_power'] = np.random.uniform(
            *self.parameter_ranges['rf_heating_power'], n_samples
        )
        
        # Total heating power
        data['total_heating_power'] = (
            data['neutral_beam_power'] + data['rf_heating_power']
        )
        
        # Heating efficiency (realistic values)
        data['heating_efficiency'] = np.random.uniform(0.7, 0.95, n_samples)
        
        return pd.DataFrame(data)
    
    def generate_fuel_systems(self, n_samples: int) -> pd.DataFrame:
        """
        Generate fuel injection and composition data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with fuel parameters
        """
        data = {}
        
        data['fuel_injection_rate'] = np.random.uniform(
            *self.parameter_ranges['fuel_injection_rate'], n_samples
        )
        
        # Deuterium-Tritium ratio (typical for D-T fusion)
        data['deuterium_fraction'] = np.random.uniform(0.4, 0.6, n_samples)
        data['tritium_fraction'] = 1.0 - data['deuterium_fraction']
        
        # Fuel purity
        data['fuel_purity'] = np.random.uniform(0.95, 0.999, n_samples)
        
        # Impurity concentration
        data['impurity_concentration'] = np.random.uniform(0.001, 0.05, n_samples)
        
        return pd.DataFrame(data)
    
    def calculate_fusion_metrics(self, plasma_df: pd.DataFrame, 
                                heating_df: pd.DataFrame, 
                                fuel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fusion performance metrics based on physical parameters.
        
        Args:
            plasma_df: Plasma parameters DataFrame
            heating_df: Heating parameters DataFrame
            fuel_df: Fuel parameters DataFrame
            
        Returns:
            DataFrame with fusion metrics
        """
        data = {}
        
        # Lawson criterion (simplified)
        data['lawson_criterion'] = (
            plasma_df['plasma_density'] * plasma_df['confinement_time']
        )
        
        # Triple product (density * temperature * confinement time)
        data['triple_product'] = (
            plasma_df['plasma_density'] * 
            plasma_df['plasma_temperature'] * 
            plasma_df['confinement_time']
        )
        
        # Fusion power (simplified Lawson criterion based)
        lawson_threshold = 1e21  # m^-3 * s
        data['fusion_power'] = np.where(
            data['lawson_criterion'] > lawson_threshold,
            (data['lawson_criterion'] / lawson_threshold) * 
            heating_df['total_heating_power'] * 
            fuel_df['fuel_purity'] * 
            (1 + np.random.normal(0, self.noise_level, len(plasma_df))),
            heating_df['total_heating_power'] * 0.1 * 
            np.random.uniform(0, 1, len(plasma_df))
        )
        
        # Q factor (fusion power / input power)
        data['q_factor'] = data['fusion_power'] / heating_df['total_heating_power']
        
        # Energy confinement time
        data['energy_confinement_time'] = (
            plasma_df['confinement_time'] * 
            (1 + np.random.normal(0, self.noise_level * 0.1, len(plasma_df)))
        )
        
        # Plasma stability indicator
        data['plasma_stability'] = np.where(
            (plasma_df['beta_plasma'] < 0.05) & 
            (plasma_df['safety_factor'] > 2.0),
            np.random.uniform(0.8, 1.0, len(plasma_df)),
            np.random.uniform(0.3, 0.8, len(plasma_df))
        )
        
        # Disruption probability
        data['disruption_probability'] = np.where(
            (plasma_df['beta_plasma'] > 0.08) | 
            (plasma_df['safety_factor'] < 1.8),
            np.random.uniform(0.3, 0.9, len(plasma_df)),
            np.random.uniform(0.01, 0.2, len(plasma_df))
        )
        
        return pd.DataFrame(data)
    
    def add_operational_status(self, n_samples: int) -> pd.DataFrame:
        """
        Add operational status and categorization.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with operational status
        """
        data = {}
        
        # Operational modes
        modes = ['startup', 'ramp_up', 'flat_top', 'ramp_down', 'shutdown']
        data['operational_mode'] = np.random.choice(modes, n_samples)
        
        # Shot duration
        data['shot_duration'] = np.random.uniform(10, 1000, n_samples)  # seconds
        
        # Time in shot
        data['time_in_shot'] = np.random.uniform(0, data['shot_duration'])
        
        # System health indicators
        data['magnet_system_health'] = np.random.uniform(0.8, 1.0, n_samples)
        data['vacuum_system_health'] = np.random.uniform(0.9, 1.0, n_samples)
        data['cooling_system_health'] = np.random.uniform(0.85, 1.0, n_samples)
        
        return pd.DataFrame(data)
    
    def generate_time_series(self, base_data: pd.DataFrame, 
                           time_steps: int = 100) -> pd.DataFrame:
        """
        Generate time series data by adding temporal variations.
        
        Args:
            base_data: Base dataset to use for time series generation
            time_steps: Number of time steps to generate
            
        Returns:
            Time series DataFrame
        """
        time_series_data = []
        
        for _, row in base_data.iterrows():
            for t in range(time_steps):
                new_row = row.copy()
                
                # Add time information
                new_row['time_step'] = t
                new_row['timestamp'] = t * 0.1  # 0.1 second intervals
                
                # Add temporal variations
                for col in row.index:
                    if col in ['plasma_temperature', 'plasma_density', 'magnetic_field']:
                        # Add sinusoidal variation with noise
                        variation = 0.05 * np.sin(2 * np.pi * t / 50) + \
                                  np.random.normal(0, self.noise_level * 0.01)
                        new_row[col] = row[col] * (1 + variation)
                
                time_series_data.append(new_row)
        
        return pd.DataFrame(time_series_data)
    
    def generate_dataset(self, n_samples: int = 10000, 
                        include_time_series: bool = False) -> pd.DataFrame:
        """
        Generate complete fusion dataset.
        
        Args:
            n_samples: Number of samples to generate
            include_time_series: Whether to include time series data
            
        Returns:
            Complete fusion dataset
        """
        # Generate component datasets
        plasma_df = self.generate_plasma_parameters(n_samples)
        heating_df = self.generate_heating_systems(n_samples)
        fuel_df = self.generate_fuel_systems(n_samples)
        fusion_metrics_df = self.calculate_fusion_metrics(plasma_df, heating_df, fuel_df)
        operational_df = self.add_operational_status(n_samples)
        
        # Combine all datasets
        complete_data = pd.concat([
            plasma_df, 
            heating_df, 
            fuel_df, 
            fusion_metrics_df, 
            operational_df
        ], axis=1)
        
        # Add sample IDs
        complete_data['sample_id'] = range(len(complete_data))
        
        # Generate time series if requested
        if include_time_series:
            complete_data = self.generate_time_series(complete_data.head(100))
        
        return complete_data
    
    def generate_anomaly_data(self, normal_data: pd.DataFrame, 
                            anomaly_fraction: float = 0.05) -> pd.DataFrame:
        """
        Generate anomalous data for testing anomaly detection algorithms.
        
        Args:
            normal_data: Normal fusion data
            anomaly_fraction: Fraction of data to make anomalous
            
        Returns:
            Data with anomalies injected
        """
        data = normal_data.copy()
        n_anomalies = int(len(data) * anomaly_fraction)
        anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
        
        # Inject different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['extreme_values', 'correlations', 'disruption'])
            
            if anomaly_type == 'extreme_values':
                # Extreme parameter values
                data.loc[idx, 'plasma_temperature'] *= np.random.uniform(2, 5)
                data.loc[idx, 'disruption_probability'] = np.random.uniform(0.8, 1.0)
                
            elif anomaly_type == 'correlations':
                # Break physical correlations
                data.loc[idx, 'pressure'] *= np.random.uniform(0.1, 0.3)
                
            elif anomaly_type == 'disruption':
                # Disruption event
                data.loc[idx, 'plasma_stability'] = np.random.uniform(0, 0.2)
                data.loc[idx, 'disruption_probability'] = np.random.uniform(0.9, 1.0)
                data.loc[idx, 'fusion_power'] *= np.random.uniform(0, 0.1)
        
        # Add anomaly labels
        data['is_anomaly'] = False
        data.loc[anomaly_indices, 'is_anomaly'] = True
        
        return data