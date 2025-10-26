"""
Unit tests for FusionDataProcessor class.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.processor import FusionDataProcessor
from src.data.generator import FusionDataGenerator


class TestFusionDataProcessor:
    """Test cases for FusionDataProcessor."""
    
    def test_initialization(self):
        """Test FusionDataProcessor initialization."""
        processor = FusionDataProcessor()
        assert processor is not None
        assert hasattr(processor, 'scaler')
        assert hasattr(processor, 'feature_selector')
        
        # Test with custom random state
        processor_seeded = FusionDataProcessor(random_state=42)
        assert processor_seeded.random_state == 42
    
    def test_validate_data_clean(self):
        """Test data validation with clean data."""
        processor = FusionDataProcessor()
        
        # Create clean test data
        data = pd.DataFrame({
            'magnetic_field': [5.0, 6.0, 4.0, 5.5],
            'plasma_current': [15.0, 18.0, 12.0, 16.0],
            'electron_density': [1.0e20, 1.2e20, 0.8e20, 1.1e20],
            'q_factor': [1.0, 1.5, 0.8, 1.2]
        })
        
        validated_data, report = processor.validate_data(data)
        
        # Should return the same data
        pd.testing.assert_frame_equal(validated_data, data)
        
        # Report should indicate clean data
        assert 'missing_values' in report
        assert 'invalid_values' in report
        assert 'duplicates' in report
        assert report['missing_values'] == 0
        assert report['invalid_values'] == 0
        assert report['duplicates'] == 0
    
    def test_validate_data_with_issues(self):
        """Test data validation with various data issues."""
        processor = FusionDataProcessor()
        
        # Create problematic data
        data = pd.DataFrame({
            'magnetic_field': [5.0, np.nan, -1.0, 5.0, 5.0],  # Missing and negative values
            'plasma_current': [15.0, 18.0, np.inf, 16.0, 15.0],  # Infinite value and duplicate
            'electron_density': [1.0e20, 1.2e20, 0.8e20, 1.1e20, 1.0e20],
            'q_factor': [1.0, 1.5, 0.8, 1.2, 1.0]  # Duplicate row
        })
        
        validated_data, report = processor.validate_data(data)
        
        # Should detect issues
        assert report['missing_values'] > 0
        assert report['invalid_values'] > 0
        assert report['duplicates'] > 0
        
        # Validated data should be smaller
        assert len(validated_data) < len(data)
        
        # Validated data should not contain missing or infinite values
        assert not validated_data.isnull().any().any()
        assert not np.isinf(validated_data.select_dtypes(include=[np.number])).any().any()
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        processor = FusionDataProcessor()
        
        # Create data with outliers and missing values
        data = pd.DataFrame({
            'magnetic_field': [5.0, 6.0, np.nan, 5.5, 100.0],  # Outlier: 100.0
            'plasma_current': [15.0, 18.0, 12.0, 16.0, 14.0],
            'electron_density': [1.0e20, 1.2e20, 0.8e20, 1.1e20, 0.9e20],
            'q_factor': [1.0, 1.5, 0.8, 1.2, 1.1]
        })
        
        cleaned_data = processor.clean_data(data)
        
        # Should remove rows with missing values and outliers
        assert len(cleaned_data) < len(data)
        assert not cleaned_data.isnull().any().any()
        
        # All values should be within reasonable ranges
        assert all(cleaned_data['magnetic_field'] < 20.0)
    
    def test_scale_features(self):
        """Test feature scaling."""
        processor = FusionDataProcessor()
        
        # Create test data with different scales
        data = pd.DataFrame({
            'small_feature': [1, 2, 3, 4, 5],
            'large_feature': [1000, 2000, 3000, 4000, 5000],
            'target': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Fit and transform
        scaled_data = processor.scale_features(data, fit=True, exclude_columns=['target'])
        
        # Check that scaling was applied
        assert not np.array_equal(data['small_feature'].values, scaled_data['small_feature'].values)
        assert not np.array_equal(data['large_feature'].values, scaled_data['large_feature'].values)
        
        # Target should remain unchanged
        np.testing.assert_array_equal(data['target'].values, scaled_data['target'].values)
        
        # Scaled features should have mean ~0 and std ~1
        assert abs(scaled_data['small_feature'].mean()) < 1e-10
        assert abs(scaled_data['large_feature'].mean()) < 1e-10
        assert abs(scaled_data['small_feature'].std() - 1.0) < 0.1
        assert abs(scaled_data['large_feature'].std() - 1.0) < 0.1
    
    def test_select_features(self):
        """Test feature selection."""
        processor = FusionDataProcessor()
        
        # Create data with some irrelevant features
        np.random.seed(42)
        n_samples = 100
        
        # Relevant features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        
        # Target correlated with x1 and x2
        target = 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n_samples)
        
        # Irrelevant features
        x3 = np.random.normal(0, 1, n_samples)  # Noise
        x4 = np.random.normal(0, 1, n_samples)  # Noise
        
        data = pd.DataFrame({
            'feature_1': x1,
            'feature_2': x2,
            'noise_1': x3,
            'noise_2': x4,
            'target': target
        })
        
        selected_data = processor.select_features(
            data, 
            target_column='target', 
            n_features=2,
            fit=True
        )
        
        # Should select only the best features
        feature_columns = [col for col in selected_data.columns if col != 'target']
        assert len(feature_columns) == 2
        
        # Should prefer the relevant features
        assert 'feature_1' in selected_data.columns or 'feature_2' in selected_data.columns
    
    def test_engineer_features(self):
        """Test feature engineering."""
        processor = FusionDataProcessor()
        
        # Create basic plasma data
        data = pd.DataFrame({
            'magnetic_field': [5.0, 6.0, 4.0],
            'plasma_current': [15.0, 18.0, 12.0],
            'electron_density': [1.0e20, 1.2e20, 0.8e20],
            'ion_temperature': [20.0, 25.0, 15.0],
            'electron_temperature': [15.0, 20.0, 12.0],
            'total_heating_power': [50.0, 60.0, 40.0]
        })
        
        engineered_data = processor.engineer_features(data)
        
        # Should have more columns than input
        assert len(engineered_data.columns) > len(data.columns)
        
        # Should contain some expected engineered features
        expected_features = [
            'thermal_pressure', 'plasma_beta', 'power_density',
            'temperature_ratio', 'normalized_density'
        ]
        
        for feature in expected_features:
            assert feature in engineered_data.columns, f"Missing engineered feature: {feature}"
        
        # Check that engineered features have reasonable values
        assert all(engineered_data['thermal_pressure'] > 0)
        assert all(engineered_data['plasma_beta'] > 0)
        assert all(engineered_data['power_density'] > 0)
    
    def test_preprocess_pipeline(self):
        """Test the complete preprocessing pipeline."""
        # Generate test data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=200)
        
        processor = FusionDataProcessor(random_state=42)
        
        # Run preprocessing pipeline
        processed_data = processor.preprocess_pipeline(
            raw_data,
            target_column='q_factor',
            test_size=0.2,
            validation_size=0.1
        )
        
        # Check that all required keys are present
        required_keys = [
            'X_train', 'X_val', 'X_test', 
            'y_train', 'y_val', 'y_test',
            'feature_names', 'preprocessing_report'
        ]
        
        for key in required_keys:
            assert key in processed_data, f"Missing key: {key}"
        
        # Check data shapes
        total_samples = len(processed_data['X_train']) + len(processed_data['X_val']) + len(processed_data['X_test'])
        assert total_samples <= len(raw_data)  # Some samples might be removed during cleaning
        
        # Check that validation set is smaller than training set
        assert len(processed_data['X_val']) < len(processed_data['X_train'])
        
        # Check that test set size is reasonable
        expected_test_size = int(0.2 * len(raw_data))
        assert abs(len(processed_data['X_test']) - expected_test_size) <= 10
        
        # Check that features are numeric and scaled
        assert processed_data['X_train'].dtype.kind in 'fc'  # Float or complex
        
        # Check that target values are preserved
        assert len(processed_data['y_train']) == len(processed_data['X_train'])
        assert len(processed_data['y_val']) == len(processed_data['X_val'])
        assert len(processed_data['y_test']) == len(processed_data['X_test'])
    
    def test_transform_new_data(self):
        """Test transforming new data with fitted processors."""
        # Generate and process training data
        generator = FusionDataGenerator(random_state=42)
        train_data = generator.generate_dataset(n_samples=100)
        
        processor = FusionDataProcessor(random_state=42)
        
        # Fit the processor
        processed_train = processor.preprocess_pipeline(
            train_data,
            target_column='q_factor',
            test_size=0.2
        )
        
        # Generate new data
        new_data = generator.generate_dataset(n_samples=50)
        
        # Transform new data using fitted processor
        # First clean and engineer features
        cleaned_new = processor.clean_data(new_data)
        engineered_new = processor.engineer_features(cleaned_new)
        
        # Remove target column for feature processing
        X_new = engineered_new.drop(columns=['q_factor'])
        
        # Scale features (without fitting)
        scaled_new = processor.scale_features(X_new, fit=False)
        
        # Select features (without fitting) 
        selected_new = processor.select_features(
            scaled_new, 
            target_column=None, 
            fit=False
        )
        
        # Should have same number of features as training data
        assert selected_new.shape[1] == processed_train['X_train'].shape[1]
    
    def test_reproducibility(self):
        """Test that preprocessing is reproducible."""
        generator = FusionDataGenerator(random_state=42)
        data = generator.generate_dataset(n_samples=100)
        
        processor1 = FusionDataProcessor(random_state=42)
        processor2 = FusionDataProcessor(random_state=42)
        
        result1 = processor1.preprocess_pipeline(data, target_column='q_factor')
        result2 = processor2.preprocess_pipeline(data, target_column='q_factor')
        
        # Results should be identical
        np.testing.assert_array_equal(result1['X_train'], result2['X_train'])
        np.testing.assert_array_equal(result1['y_train'], result2['y_train'])
        np.testing.assert_array_equal(result1['X_test'], result2['X_test'])
        np.testing.assert_array_equal(result1['y_test'], result2['y_test'])
    
    def test_handle_edge_cases(self):
        """Test handling of edge cases."""
        processor = FusionDataProcessor()
        
        # Test with very small dataset
        small_data = pd.DataFrame({
            'feature_1': [1, 2],
            'feature_2': [3, 4],
            'target': [0.1, 0.2]
        })
        
        # Should handle small datasets gracefully
        try:
            result = processor.preprocess_pipeline(
                small_data, 
                target_column='target',
                test_size=0.5
            )
            # If it succeeds, check basic structure
            assert 'X_train' in result
            assert 'y_train' in result
        except ValueError:
            # It's acceptable to raise an error for very small datasets
            pass
        
        # Test with single column
        single_col_data = pd.DataFrame({
            'only_feature': [1, 2, 3, 4, 5],
            'target': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = processor.preprocess_pipeline(
            single_col_data,
            target_column='target'
        )
        
        # Should work with single feature
        assert result['X_train'].shape[1] >= 1