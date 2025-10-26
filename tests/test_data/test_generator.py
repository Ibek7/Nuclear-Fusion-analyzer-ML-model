"""
Unit tests for FusionDataGenerator class.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.generator import FusionDataGenerator


class TestFusionDataGenerator:
    """Test cases for FusionDataGenerator."""
    
    def test_initialization(self):
        """Test FusionDataGenerator initialization."""
        generator = FusionDataGenerator()
        assert generator is not None
        assert hasattr(generator, 'tokamak_params')
        assert hasattr(generator, 'plasma_params')
        
        # Test with custom random state
        generator_seeded = FusionDataGenerator(random_state=42)
        assert generator_seeded.random_state == 42
    
    def test_generate_plasma_parameters(self):
        """Test plasma parameter generation."""
        generator = FusionDataGenerator(random_state=42)
        params = generator._generate_plasma_parameters(n_samples=10)
        
        # Check if all required columns are present
        expected_columns = [
            'magnetic_field', 'plasma_current', 'electron_density',
            'ion_temperature', 'electron_temperature', 'plasma_pressure',
            'beta_normalized', 'safety_factor'
        ]
        
        for col in expected_columns:
            assert col in params.columns, f"Missing column: {col}"
        
        # Check data types and values
        assert len(params) == 10
        assert params['magnetic_field'].dtype == 'float64'
        assert all(params['magnetic_field'] > 0), "Magnetic field should be positive"
        assert all(params['electron_density'] > 0), "Electron density should be positive"
        assert all(params['ion_temperature'] > 0), "Ion temperature should be positive"
    
    def test_generate_heating_systems(self):
        """Test heating system parameter generation."""
        generator = FusionDataGenerator(random_state=42)
        heating = generator._generate_heating_systems(n_samples=10)
        
        expected_columns = [
            'neutral_beam_power', 'rf_heating_power', 'ohmic_heating_power',
            'total_heating_power', 'heating_efficiency'
        ]
        
        for col in expected_columns:
            assert col in heating.columns, f"Missing column: {col}"
        
        # Check physical constraints
        assert all(heating['total_heating_power'] >= 0), "Total heating power should be non-negative"
        assert all(heating['heating_efficiency'] > 0), "Heating efficiency should be positive"
        assert all(heating['heating_efficiency'] <= 1), "Heating efficiency should be <= 1"
    
    def test_generate_fuel_systems(self):
        """Test fuel system parameter generation."""
        generator = FusionDataGenerator(random_state=42)
        fuel = generator._generate_fuel_systems(n_samples=10)
        
        expected_columns = [
            'fuel_density', 'deuterium_fraction', 'tritium_fraction',
            'impurity_concentration', 'fueling_rate'
        ]
        
        for col in expected_columns:
            assert col in fuel.columns, f"Missing column: {col}"
        
        # Check fuel fraction constraints
        assert all(fuel['deuterium_fraction'] >= 0), "Deuterium fraction should be non-negative"
        assert all(fuel['tritium_fraction'] >= 0), "Tritium fraction should be non-negative"
        assert all(fuel['deuterium_fraction'] <= 1), "Deuterium fraction should be <= 1"
        assert all(fuel['tritium_fraction'] <= 1), "Tritium fraction should be <= 1"
        
        # Check that D+T fractions are reasonable
        fuel_sum = fuel['deuterium_fraction'] + fuel['tritium_fraction']
        assert all(fuel_sum <= 1.1), "D+T fractions should sum to <= 1 (allowing small numerical errors)"
    
    def test_generate_confinement_metrics(self):
        """Test confinement metrics calculation."""
        generator = FusionDataGenerator(random_state=42)
        
        # Create sample plasma data
        plasma_data = pd.DataFrame({
            'magnetic_field': [5.0, 6.0, 4.0],
            'plasma_current': [15.0, 18.0, 12.0],
            'electron_density': [1.0e20, 1.2e20, 0.8e20],
            'ion_temperature': [20.0, 25.0, 15.0],
            'electron_temperature': [15.0, 20.0, 12.0],
            'total_heating_power': [50.0, 60.0, 40.0]
        })
        
        confinement = generator._calculate_confinement_metrics(plasma_data)
        
        expected_columns = [
            'confinement_time', 'energy_confinement_time',
            'triple_product', 'lawson_criterion'
        ]
        
        for col in expected_columns:
            assert col in confinement.columns, f"Missing column: {col}"
        
        # Check physical constraints
        assert all(confinement['confinement_time'] > 0), "Confinement time should be positive"
        assert all(confinement['triple_product'] > 0), "Triple product should be positive"
    
    def test_calculate_q_factor(self):
        """Test Q factor calculation."""
        generator = FusionDataGenerator(random_state=42)
        
        # Create sample input data
        input_data = pd.DataFrame({
            'magnetic_field': [5.0, 6.0, 4.0],
            'plasma_current': [15.0, 18.0, 12.0],
            'electron_density': [1.0e20, 1.2e20, 0.8e20],
            'ion_temperature': [20.0, 25.0, 15.0],
            'electron_temperature': [15.0, 20.0, 12.0],
            'total_heating_power': [50.0, 60.0, 40.0],
            'fuel_density': [5.0e19, 6.0e19, 4.0e19],
            'deuterium_fraction': [0.5, 0.5, 0.5],
            'tritium_fraction': [0.5, 0.5, 0.5],
            'confinement_time': [3.0, 3.5, 2.5]
        })
        
        q_factors = generator._calculate_q_factor(input_data)
        
        assert len(q_factors) == len(input_data)
        assert all(q_factors >= 0), "Q factor should be non-negative"
        assert all(np.isfinite(q_factors)), "Q factor should be finite"
    
    def test_inject_anomalies(self):
        """Test anomaly injection."""
        generator = FusionDataGenerator(random_state=42)
        
        # Create clean data
        data = pd.DataFrame({
            'magnetic_field': [5.0] * 100,
            'plasma_current': [15.0] * 100,
            'q_factor': [1.0] * 100
        })
        
        # Inject anomalies
        anomalous_data = generator._inject_anomalies(data.copy(), anomaly_rate=0.1)
        
        # Check that some data has changed (anomalies injected)
        assert not data.equals(anomalous_data), "Data should be modified after anomaly injection"
        
        # Check that we still have the same number of rows
        assert len(anomalous_data) == len(data)
        
        # Check that roughly the expected fraction has anomalies
        # (This is probabilistic, so we allow some variance)
        different_rows = (data != anomalous_data).any(axis=1).sum()
        expected_anomalies = int(0.1 * len(data))
        assert abs(different_rows - expected_anomalies) <= 5, "Anomaly rate should be approximately correct"
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = FusionDataGenerator(random_state=42)
        dataset = generator.generate_dataset(n_samples=50)
        
        # Check basic properties
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) == 50
        assert 'q_factor' in dataset.columns
        
        # Check that all columns have numeric data
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns
        assert len(numeric_columns) > 10, "Should have multiple numeric columns"
        
        # Check for no missing values
        assert not dataset.isnull().any().any(), "Dataset should not contain missing values"
        
        # Check Q factor range is reasonable
        assert dataset['q_factor'].min() >= 0, "Q factor should be non-negative"
        assert dataset['q_factor'].max() <= 100, "Q factor should be reasonable (< 100)"
    
    def test_generate_time_series(self):
        """Test time series generation."""
        generator = FusionDataGenerator(random_state=42)
        time_series = generator.generate_time_series(
            n_timepoints=100,
            dt=0.1,
            base_params={'magnetic_field': 5.0, 'plasma_current': 15.0}
        )
        
        # Check basic properties
        assert isinstance(time_series, pd.DataFrame)
        assert len(time_series) == 100
        assert 'time' in time_series.columns
        assert 'q_factor' in time_series.columns
        
        # Check time column
        expected_times = np.arange(0, 10.0, 0.1)
        np.testing.assert_array_almost_equal(time_series['time'].values, expected_times)
        
        # Check that magnetic field varies around base value
        mag_field_mean = time_series['magnetic_field'].mean()
        assert abs(mag_field_mean - 5.0) < 1.0, "Magnetic field should vary around base value"
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        generator1 = FusionDataGenerator(random_state=42)
        generator2 = FusionDataGenerator(random_state=42)
        
        dataset1 = generator1.generate_dataset(n_samples=20)
        dataset2 = generator2.generate_dataset(n_samples=20)
        
        # Should be identical
        pd.testing.assert_frame_equal(dataset1, dataset2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different random states produce different data."""
        generator1 = FusionDataGenerator(random_state=42)
        generator2 = FusionDataGenerator(random_state=123)
        
        dataset1 = generator1.generate_dataset(n_samples=20)
        dataset2 = generator2.generate_dataset(n_samples=20)
        
        # Should be different
        assert not dataset1.equals(dataset2), "Different random states should produce different data"
    
    def test_parameter_bounds(self):
        """Test that generated parameters are within physical bounds."""
        generator = FusionDataGenerator(random_state=42)
        dataset = generator.generate_dataset(n_samples=100)
        
        # Test some key physical constraints
        assert all(dataset['magnetic_field'] > 0), "Magnetic field must be positive"
        assert all(dataset['magnetic_field'] < 20), "Magnetic field should be reasonable (< 20 T)"
        
        assert all(dataset['electron_density'] > 0), "Electron density must be positive"
        assert all(dataset['electron_density'] < 1e22), "Electron density should be reasonable"
        
        assert all(dataset['ion_temperature'] > 0), "Ion temperature must be positive"
        assert all(dataset['ion_temperature'] < 100), "Ion temperature should be reasonable (< 100 keV)"
        
        if 'deuterium_fraction' in dataset.columns:
            assert all(dataset['deuterium_fraction'] >= 0), "Deuterium fraction must be non-negative"
            assert all(dataset['deuterium_fraction'] <= 1), "Deuterium fraction must be <= 1"