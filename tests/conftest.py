"""
Test configuration and fixtures for Nuclear Fusion Analyzer tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.generator import FusionDataGenerator
from src.data.processor import FusionDataProcessor


@pytest.fixture
def fusion_generator():
    """Fixture for FusionDataGenerator."""
    return FusionDataGenerator(random_state=42)


@pytest.fixture
def fusion_processor():
    """Fixture for FusionDataProcessor."""
    return FusionDataProcessor(random_state=42)


@pytest.fixture
def sample_fusion_data():
    """Fixture for small sample of fusion data."""
    generator = FusionDataGenerator(random_state=42)
    return generator.generate_dataset(n_samples=100)


@pytest.fixture
def processed_fusion_data(sample_fusion_data):
    """Fixture for preprocessed fusion data."""
    processor = FusionDataProcessor(random_state=42)
    return processor.preprocess_pipeline(
        sample_fusion_data, 
        target_column='q_factor',
        test_size=0.2,
        validation_size=0.1
    )


@pytest.fixture
def mock_plasma_params():
    """Fixture for mock plasma parameters."""
    return {
        'magnetic_field': 5.3,
        'plasma_current': 15.0,
        'electron_density': 1.0e20,
        'ion_temperature': 20.0,
        'electron_temperature': 15.0,
        'fuel_density': 5.0e19,
        'deuterium_fraction': 0.5,
        'tritium_fraction': 0.5,
        'heating_power': 50.0,
        'confinement_time': 3.0
    }


class TestDataHelper:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_test_dataframe(n_samples=50):
        """Create a test DataFrame with fusion parameters."""
        np.random.seed(42)
        
        data = {
            'magnetic_field': np.random.normal(5.0, 1.0, n_samples),
            'plasma_current': np.random.normal(15.0, 3.0, n_samples),
            'electron_density': np.random.normal(1.0e20, 2.0e19, n_samples),
            'ion_temperature': np.random.normal(20.0, 5.0, n_samples),
            'electron_temperature': np.random.normal(15.0, 3.0, n_samples),
            'heating_power': np.random.normal(50.0, 10.0, n_samples),
            'confinement_time': np.random.normal(3.0, 0.5, n_samples),
            'q_factor': np.random.normal(1.5, 0.5, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_invalid_dataframe():
        """Create a DataFrame with invalid/missing data for testing."""
        data = {
            'magnetic_field': [5.0, np.nan, -1.0, 10.0],
            'plasma_current': [15.0, 20.0, np.inf, 5.0],
            'electron_density': [1.0e20, 2.0e20, -1.0e19, 1.5e20],
            'q_factor': [1.0, np.nan, 2.0, 0.5]
        }
        
        return pd.DataFrame(data)


# Constants for testing
TEST_TOLERANCE = 1e-10
RANDOM_SEED = 42
SMALL_DATASET_SIZE = 100
MEDIUM_DATASET_SIZE = 1000