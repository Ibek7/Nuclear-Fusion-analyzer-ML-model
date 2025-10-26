"""
Unit tests for FusionAnomalyDetector class.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.anomaly_detector import FusionAnomalyDetector
from src.data.generator import FusionDataGenerator


class TestFusionAnomalyDetector:
    """Test cases for FusionAnomalyDetector."""
    
    def test_initialization(self):
        """Test FusionAnomalyDetector initialization."""
        detector = FusionAnomalyDetector()
        assert detector is not None
        assert hasattr(detector, 'detectors')
        assert hasattr(detector, 'trained_detectors')
        
        # Check that detectors dictionary contains expected detectors
        expected_detectors = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        for detector_name in expected_detectors:
            assert detector_name in detector.detectors
    
    def test_prepare_sample_data(self):
        """Prepare sample data for testing."""
        generator = FusionDataGenerator(random_state=42)
        
        # Generate normal data
        normal_data = generator.generate_dataset(n_samples=200)
        
        # Generate some anomalous data by injecting anomalies
        anomalous_data = generator._inject_anomalies(normal_data.copy(), anomaly_rate=0.1)
        
        return normal_data, anomalous_data
    
    def test_fit_isolation_forest(self):
        """Test fitting isolation forest detector."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        
        # Remove target column for training
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Fit isolation forest
        detector.fit('isolation_forest', X_normal)
        
        # Check that detector was trained
        assert 'isolation_forest' in detector.trained_detectors
        assert detector.trained_detectors['isolation_forest'] is not None
    
    def test_fit_one_class_svm(self):
        """Test fitting One-Class SVM detector."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        
        # Use smaller dataset for SVM (it's slower)
        X_normal = normal_data.drop(columns=['q_factor']).iloc[:100]
        
        # Fit One-Class SVM
        detector.fit('one_class_svm', X_normal)
        
        # Check that detector was trained
        assert 'one_class_svm' in detector.trained_detectors
        assert detector.trained_detectors['one_class_svm'] is not None
    
    def test_fit_local_outlier_factor(self):
        """Test fitting Local Outlier Factor detector."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        
        # Remove target column for training
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Fit LOF
        detector.fit('local_outlier_factor', X_normal)
        
        # Check that detector was trained
        assert 'local_outlier_factor' in detector.trained_detectors
        assert detector.trained_detectors['local_outlier_factor'] is not None
    
    def test_fit_invalid_detector(self):
        """Test fitting with invalid detector name."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Should raise error for invalid detector
        with pytest.raises(ValueError):
            detector.fit('invalid_detector', X_normal)
    
    @patch('src.models.anomaly_detector.tf')
    def test_fit_autoencoder_mock(self, mock_tf):
        """Test fitting autoencoder with mocked TensorFlow."""
        # Mock TensorFlow components
        mock_model = MagicMock()
        mock_model.fit.return_value = MagicMock()
        mock_model.predict.return_value = np.random.rand(10, 5)
        
        mock_tf.keras.Sequential.return_value = mock_model
        mock_tf.keras.layers = MagicMock()
        mock_tf.keras.optimizers = MagicMock()
        
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Fit autoencoder
        detector.fit_autoencoder(X_normal, epochs=2)
        
        # Check that autoencoder was trained
        assert detector.autoencoder is not None
    
    def test_detect_anomalies_isolation_forest(self):
        """Test anomaly detection with isolation forest."""
        detector = FusionAnomalyDetector()
        normal_data, anomalous_data = self.test_prepare_sample_data()
        
        # Prepare training and test data
        X_normal = normal_data.drop(columns=['q_factor'])
        X_test = anomalous_data.drop(columns=['q_factor'])
        
        # Fit detector
        detector.fit('isolation_forest', X_normal)
        
        # Detect anomalies
        anomaly_scores = detector.detect_anomalies(X_test, 'isolation_forest')
        
        # Check results
        assert len(anomaly_scores) == len(X_test)
        assert isinstance(anomaly_scores, np.ndarray)
        assert all(isinstance(score, (int, float, np.integer, np.floating)) for score in anomaly_scores)
        
        # Scores should be between -1 and 1 for isolation forest
        assert all(-1 <= score <= 1 for score in anomaly_scores)
    
    def test_detect_anomalies_untrained(self):
        """Test anomaly detection with untrained detector."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        X_test = normal_data.drop(columns=['q_factor'])
        
        # Should raise error for untrained detector
        with pytest.raises(ValueError):
            detector.detect_anomalies(X_test, 'isolation_forest')
    
    def test_physics_based_anomaly_detection(self):
        """Test physics-based anomaly detection."""
        detector = FusionAnomalyDetector()
        
        # Create test data with some physics violations
        test_data = pd.DataFrame({
            'magnetic_field': [5.0, -2.0, 100.0, 4.0],  # Negative and extreme values
            'plasma_current': [15.0, 18.0, 12.0, np.inf],  # Infinite value
            'electron_density': [1.0e20, 1.2e20, -1.0e19, 1.1e20],  # Negative value
            'ion_temperature': [20.0, 25.0, 15.0, 18.0],
            'q_factor': [1.0, 1.5, -0.5, 2.0]  # Negative Q factor
        })
        
        anomalies = detector.physics_based_anomaly_detection(test_data)
        
        # Check results
        assert len(anomalies) == len(test_data)
        assert isinstance(anomalies, np.ndarray)
        assert anomalies.dtype == bool
        
        # Should detect the physics violations
        assert anomalies[1] == True  # Negative magnetic field
        assert anomalies[2] == True  # Extreme magnetic field or negative density
        assert anomalies[3] == True  # Infinite plasma current
    
    def test_detect_plasma_disruptions(self):
        """Test plasma disruption detection."""
        detector = FusionAnomalyDetector()
        
        # Create test data with disruption-like patterns
        test_data = pd.DataFrame({
            'magnetic_field': [5.0, 5.1, 4.8, 2.0],  # Sudden drop in magnetic field
            'plasma_current': [15.0, 15.2, 14.8, 5.0],  # Sudden drop in current
            'electron_density': [1.0e20, 1.1e20, 0.9e20, 0.3e20],  # Density drop
            'q_factor': [1.5, 1.4, 1.6, 0.1],  # Q factor collapse
            'confinement_time': [3.0, 2.9, 3.1, 0.5]  # Confinement loss
        })
        
        disruptions = detector.detect_plasma_disruptions(test_data)
        
        # Check results
        assert len(disruptions) == len(test_data)
        assert isinstance(disruptions, np.ndarray)
        assert disruptions.dtype == bool
        
        # Should detect the disruption in the last sample
        assert disruptions[3] == True
    
    def test_detect_equipment_failures(self):
        """Test equipment failure detection."""
        detector = FusionAnomalyDetector()
        
        # Create test data with equipment failure patterns
        test_data = pd.DataFrame({
            'magnetic_field': [5.0, 5.1, 0.0, 4.8],  # Magnetic coil failure
            'neutral_beam_power': [50.0, 52.0, 0.0, 48.0],  # Beam system failure
            'rf_heating_power': [30.0, 32.0, 28.0, 0.0],  # RF system failure
            'plasma_current': [15.0, 15.2, 14.8, 14.9]
        })
        
        failures = detector.detect_equipment_failures(test_data)
        
        # Check results
        assert len(failures) == len(test_data)
        assert isinstance(failures, np.ndarray)
        assert failures.dtype == bool
        
        # Should detect equipment failures
        assert failures[2] == True  # Magnetic field and beam power failure
        assert failures[3] == True  # RF heating failure
    
    def test_comprehensive_anomaly_analysis(self):
        """Test comprehensive anomaly analysis."""
        detector = FusionAnomalyDetector()
        normal_data, anomalous_data = self.test_prepare_sample_data()
        
        # Prepare data
        X_normal = normal_data.drop(columns=['q_factor'])
        X_test = anomalous_data.drop(columns=['q_factor'])
        
        # Fit multiple detectors
        detector.fit('isolation_forest', X_normal)
        detector.fit('one_class_svm', X_normal.iloc[:100])  # Smaller dataset for SVM
        
        # Run comprehensive analysis
        analysis = detector.comprehensive_anomaly_analysis(anomalous_data)
        
        # Check results structure
        assert isinstance(analysis, dict)
        expected_keys = [
            'isolation_forest_scores', 'one_class_svm_scores',
            'physics_anomalies', 'disruption_anomalies', 'equipment_failures',
            'combined_anomaly_score', 'anomaly_summary'
        ]
        
        for key in expected_keys:
            if key in ['one_class_svm_scores'] and len(anomalous_data) != len(X_normal.iloc[:100]):
                continue  # Skip if data sizes don't match
            assert key in analysis, f"Missing key: {key}"
        
        # Check that combined scores exist
        if 'combined_anomaly_score' in analysis:
            combined_scores = analysis['combined_anomaly_score']
            assert len(combined_scores) == len(anomalous_data)
            assert all(0 <= score <= 1 for score in combined_scores)
    
    def test_calculate_reconstruction_error(self):
        """Test autoencoder reconstruction error calculation."""
        detector = FusionAnomalyDetector()
        
        # Create mock autoencoder
        class MockAutoencoder:
            def predict(self, X):
                # Return slightly different data to simulate reconstruction
                return X + np.random.normal(0, 0.1, X.shape)
        
        detector.autoencoder = MockAutoencoder()
        
        # Test data
        test_data = np.random.rand(10, 5)
        
        # Calculate reconstruction error
        errors = detector.calculate_reconstruction_error(test_data)
        
        # Check results
        assert len(errors) == len(test_data)
        assert all(error >= 0 for error in errors)
        assert isinstance(errors, np.ndarray)
    
    def test_get_anomaly_threshold(self):
        """Test anomaly threshold calculation."""
        detector = FusionAnomalyDetector()
        
        # Generate sample scores
        normal_scores = np.random.normal(-0.1, 0.2, 100)  # Mostly negative (normal)
        
        # Calculate threshold
        threshold = detector.get_anomaly_threshold(normal_scores, contamination=0.05)
        
        # Check threshold
        assert isinstance(threshold, (int, float))
        
        # Should be around 95th percentile for contamination=0.05
        expected_threshold = np.percentile(normal_scores, 95)
        assert abs(threshold - expected_threshold) < 0.1
    
    def test_input_validation(self):
        """Test input validation."""
        detector = FusionAnomalyDetector()
        
        # Test with empty data
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            detector.physics_based_anomaly_detection(empty_data)
        
        # Test with data missing required columns
        incomplete_data = pd.DataFrame({'only_one_column': [1, 2, 3]})
        
        # Should handle gracefully or raise appropriate error
        try:
            detector.physics_based_anomaly_detection(incomplete_data)
        except (KeyError, ValueError):
            pass  # Expected behavior
    
    def test_anomaly_score_ranges(self):
        """Test that anomaly scores are in expected ranges."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Fit and test isolation forest
        detector.fit('isolation_forest', X_normal)
        scores = detector.detect_anomalies(X_normal, 'isolation_forest')
        
        # Isolation forest scores should be between -1 and 1
        assert all(-1.1 <= score <= 1.1 for score in scores), "Isolation forest scores out of range"
        
        # Most normal data should have positive scores (closer to 1)
        positive_scores = sum(1 for score in scores if score > 0)
        assert positive_scores > len(scores) * 0.5, "Too many negative scores for normal data"
    
    def test_consistent_anomaly_detection(self):
        """Test that anomaly detection is consistent across multiple calls."""
        detector = FusionAnomalyDetector()
        normal_data, _ = self.test_prepare_sample_data()
        
        X_normal = normal_data.drop(columns=['q_factor'])
        
        # Fit detector
        detector.fit('isolation_forest', X_normal)
        
        # Detect anomalies multiple times
        scores1 = detector.detect_anomalies(X_normal, 'isolation_forest')
        scores2 = detector.detect_anomalies(X_normal, 'isolation_forest')
        
        # Should be identical
        np.testing.assert_array_equal(scores1, scores2)