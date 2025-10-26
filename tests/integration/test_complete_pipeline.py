"""
Integration tests for the Nuclear Fusion Analyzer ML model.

These tests verify the end-to-end functionality of the complete system,
including data flow, model training, prediction pipelines, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.data.generator import FusionDataGenerator
from src.data.processor import FusionDataProcessor
from src.models.fusion_predictor import FusionPredictor
from src.models.anomaly_detector import FusionAnomalyDetector
from src.utils.config_manager import ConfigManager
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.data_validator import FusionDataValidator
from src.utils.hyperparameter_optimizer import HyperparameterOptimizer
from src.utils.model_interpreter import ModelInterpreter
from src.utils.advanced_anomaly_detector import FusionAnomalyDetector as AdvancedAnomalyDetector
from src.visualization.advanced_plots import FusionVisualizationSuite


class TestEndToEndPipeline:
    """Test the complete end-to-end ML pipeline."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            'data_generation': {
                'n_samples': 1000,
                'noise_level': 0.1,
                'random_state': 42
            },
            'model_training': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 3
            },
            'performance_monitoring': {
                'enable_gpu_monitoring': False,
                'log_interval': 1.0
            }
        }
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_complete_data_pipeline(self, sample_config, temp_directory):
        """Test the complete data generation and processing pipeline."""
        # Step 1: Generate data
        generator = FusionDataGenerator(
            random_state=sample_config['data_generation']['random_state']
        )
        
        raw_data = generator.generate_dataset(
            n_samples=sample_config['data_generation']['n_samples'],
            noise_level=sample_config['data_generation']['noise_level']
        )
        
        assert len(raw_data) == sample_config['data_generation']['n_samples']
        assert 'plasma_current' in raw_data.columns
        assert 'q_factor' in raw_data.columns
        
        # Step 2: Process data
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        
        assert processed_data is not None
        assert len(processed_data) <= len(raw_data)  # May remove outliers
        
        # Step 3: Validate data
        validator = FusionDataValidator()
        validation_result = validator.validate_dataset(processed_data)
        
        assert validation_result.is_valid
        assert validation_result.physics_constraints_passed
        
        # Step 4: Feature engineering
        X, y = processor.prepare_features(
            processed_data, 
            target_column='q_factor'
        )
        
        assert X.shape[0] == len(processed_data)
        assert len(y) == len(processed_data)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_complete_model_training_pipeline(self, sample_config, temp_directory):
        """Test the complete model training and evaluation pipeline."""
        # Generate and prepare data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=1000)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        # Initialize predictor
        predictor = FusionPredictor()
        
        # Train multiple models
        model_types = ['random_forest', 'gradient_boosting', 'neural_network']
        
        for model_type in model_types:
            # Train model
            model = predictor.train_model(X, y, model_type=model_type)
            assert model is not None
            
            # Make predictions
            predictions = predictor.predict(X[:10])
            assert len(predictions) == 10
            assert not np.isnan(predictions).any()
            
            # Evaluate model
            metrics = predictor.evaluate_model(X, y)
            assert 'mse' in metrics
            assert 'r2' in metrics
            assert metrics['mse'] >= 0
            assert metrics['r2'] <= 1
        
        # Test ensemble prediction
        ensemble_predictions = predictor.predict_ensemble(X[:10])
        assert len(ensemble_predictions) == 10
        assert not np.isnan(ensemble_predictions).any()
    
    def test_anomaly_detection_pipeline(self, sample_config, temp_directory):
        """Test the complete anomaly detection pipeline."""
        # Generate normal and anomalous data
        generator = FusionDataGenerator(random_state=42)
        normal_data = generator.generate_dataset(n_samples=800)
        
        # Create anomalous data by modifying some parameters
        anomalous_data = normal_data.copy()
        anomalous_indices = np.random.choice(len(anomalous_data), size=50, replace=False)
        anomalous_data.loc[anomalous_indices, 'plasma_current'] *= 10  # Extreme values
        
        combined_data = pd.concat([normal_data, anomalous_data.iloc[anomalous_indices]])
        
        # Basic anomaly detection
        basic_detector = FusionAnomalyDetector()
        basic_detector.fit(normal_data.drop(columns=['timestamp'] if 'timestamp' in normal_data.columns else []))
        
        anomaly_scores = basic_detector.detect_anomalies(
            combined_data.drop(columns=['timestamp'] if 'timestamp' in combined_data.columns else [])
        )
        
        assert len(anomaly_scores) == len(combined_data)
        assert np.all(anomaly_scores >= 0)
        assert np.all(anomaly_scores <= 1)
        
        # Advanced anomaly detection
        advanced_detector = AdvancedAnomalyDetector()
        results = advanced_detector.detect_comprehensive(combined_data)
        
        assert 'isolation_forest' in results
        assert hasattr(results['isolation_forest'], 'anomaly_scores')
        assert hasattr(results['isolation_forest'], 'anomaly_labels')
    
    def test_hyperparameter_optimization_pipeline(self, sample_config, temp_directory):
        """Test the hyperparameter optimization pipeline."""
        # Generate data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=500)  # Smaller for faster testing
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer()
        
        # Test different optimization strategies
        strategies = ['grid_search', 'random_search']
        
        for strategy in strategies:
            result = optimizer.optimize_model(
                X, y, 
                model_type='random_forest',
                strategy=strategy,
                n_trials=5  # Small number for testing
            )
            
            assert result.best_score is not None
            assert result.best_params is not None
            assert result.optimization_history is not None
            assert len(result.optimization_history) <= 5
    
    def test_model_interpretation_pipeline(self, sample_config, temp_directory):
        """Test the model interpretation pipeline."""
        # Generate and prepare data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=500)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        # Train a model
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Initialize interpreter
        feature_names = processor.get_feature_names()
        interpreter = ModelInterpreter(model, feature_names)
        
        # Test feature importance calculation
        importance = interpreter.calculate_feature_importance(X, y, method='tree_based')
        
        assert len(importance.feature_names) == X.shape[1]
        assert len(importance.importance_scores) == X.shape[1]
        assert np.all(importance.importance_scores >= 0)
        
        # Test comprehensive interpretation report
        report = interpreter.generate_interpretation_report(X, y, sample_instance_idx=0)
        
        assert 'feature_importance' in report
        assert 'physics_insights' in report
    
    def test_performance_monitoring_pipeline(self, sample_config, temp_directory):
        """Test the performance monitoring pipeline."""
        monitor = PerformanceMonitor(config=sample_config['performance_monitoring'])
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some computation
        generator = FusionDataGenerator(random_state=42)
        data = generator.generate_dataset(n_samples=100)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(data)
        
        # Record execution time
        with monitor.timer("data_processing"):
            time.sleep(0.1)  # Simulate processing time
        
        # Stop monitoring and get report
        monitor.stop_monitoring()
        report = monitor.generate_report()
        
        assert 'execution_times' in report
        assert 'system_metrics' in report
        assert 'data_processing' in report['execution_times']
    
    def test_visualization_pipeline(self, sample_config, temp_directory):
        """Test the visualization pipeline."""
        # Generate data
        generator = FusionDataGenerator(random_state=42)
        data = generator.generate_dataset(n_samples=200)
        
        # Initialize visualization suite
        viz_suite = FusionVisualizationSuite()
        
        # Test time series plotting
        fig = viz_suite.plot_plasma_parameters_time_series(
            data, 
            parameters=['plasma_current', 'magnetic_field'],
            interactive=False  # Use matplotlib for testing
        )
        
        assert fig is not None
        
        # Test correlation matrix
        correlation_fig = viz_suite.plot_parameter_correlation_matrix(
            data.select_dtypes(include=[np.number]),
            interactive=False
        )
        
        assert correlation_fig is not None
        
        # Save plots to verify they work
        output_path = os.path.join(temp_directory, 'test_plot.png')
        viz_suite.save_plot(fig, output_path)
        assert os.path.exists(output_path)


class TestAPIIntegration:
    """Test API integration and endpoints."""
    
    @pytest.fixture
    def api_client(self):
        """Create a test API client."""
        from api.app import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """Test the health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_prediction_endpoint(self, api_client):
        """Test the prediction endpoint."""
        # Sample input data
        test_data = {
            "magnetic_field": 5.3,
            "plasma_current": 15.0,
            "electron_density": 1.0e20,
            "ion_temperature": 20.0,
            "electron_temperature": 15.0,
            "neutral_beam_power": 50.0,
            "rf_heating_power": 30.0
        }
        
        response = api_client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert "q_factor" in result["predictions"]
        assert isinstance(result["predictions"]["q_factor"], (int, float))
    
    def test_batch_prediction_endpoint(self, api_client):
        """Test the batch prediction endpoint."""
        # Sample batch data
        test_batch = [
            {
                "magnetic_field": 5.3,
                "plasma_current": 15.0,
                "electron_density": 1.0e20,
                "ion_temperature": 20.0,
                "electron_temperature": 15.0,
                "neutral_beam_power": 50.0,
                "rf_heating_power": 30.0
            },
            {
                "magnetic_field": 4.8,
                "plasma_current": 12.0,
                "electron_density": 8.0e19,
                "ion_temperature": 18.0,
                "electron_temperature": 14.0,
                "neutral_beam_power": 45.0,
                "rf_heating_power": 25.0
            }
        ]
        
        response = api_client.post("/predict/batch", json={"samples": test_batch})
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 2
        
        for prediction in result["predictions"]:
            assert "q_factor" in prediction
    
    def test_anomaly_detection_endpoint(self, api_client):
        """Test the anomaly detection endpoint."""
        test_data = {
            "magnetic_field": 5.3,
            "plasma_current": 15.0,
            "electron_density": 1.0e20,
            "ion_temperature": 20.0,
            "electron_temperature": 15.0,
            "neutral_beam_power": 50.0,
            "rf_heating_power": 30.0
        }
        
        response = api_client.post("/detect-anomaly", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert isinstance(result["anomaly_score"], (int, float))
        assert isinstance(result["is_anomaly"], bool)


class TestDataFlowIntegration:
    """Test data flow between different components."""
    
    def test_data_generator_to_processor_flow(self):
        """Test data flow from generator to processor."""
        # Generate data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=100)
        
        # Process data
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        
        # Verify data integrity
        assert len(processed_data) <= len(raw_data)
        assert processed_data.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        
        # Check that essential columns are preserved
        essential_columns = ['plasma_current', 'magnetic_field', 'q_factor']
        for col in essential_columns:
            if col in raw_data.columns:
                assert col in processed_data.columns
    
    def test_processor_to_model_flow(self):
        """Test data flow from processor to model."""
        # Generate and process data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=200)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        # Train model
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Verify predictions work
        predictions = predictor.predict(X[:10])
        assert len(predictions) == 10
        assert not np.isnan(predictions).any()
        
        # Verify prediction shapes match
        assert predictions.shape == (10,)
    
    def test_model_to_interpretation_flow(self):
        """Test data flow from model to interpretation."""
        # Generate data and train model
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=150)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Interpret model
        feature_names = processor.get_feature_names()
        interpreter = ModelInterpreter(model, feature_names)
        
        importance = interpreter.calculate_feature_importance(X, y, method='tree_based')
        
        # Verify interpretation results
        assert len(importance.feature_names) == len(feature_names)
        assert len(importance.importance_scores) == len(feature_names)
        assert sum(importance.importance_scores) > 0  # At least some features should be important


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        processor = FusionDataProcessor()
        
        # Test with NaN values
        invalid_data = pd.DataFrame({
            'plasma_current': [1.0, np.nan, 3.0],
            'magnetic_field': [5.0, 6.0, np.nan],
            'q_factor': [1.5, 2.0, 1.8]
        })
        
        processed_data = processor.preprocess(invalid_data)
        
        # Should handle NaN values appropriately
        assert not processed_data.isna().any().any()
    
    def test_model_training_with_insufficient_data(self):
        """Test model training with insufficient data."""
        # Create very small dataset
        X = np.random.random((5, 3))
        y = np.random.random(5)
        
        predictor = FusionPredictor()
        
        # Should handle small datasets gracefully
        with pytest.warns(UserWarning):
            model = predictor.train_model(X, y, model_type='random_forest')
            assert model is not None
    
    def test_prediction_with_out_of_range_values(self):
        """Test predictions with out-of-range input values."""
        # Train model with normal data
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=100)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Create out-of-range test data
        extreme_data = X.copy()
        extreme_data[:, 0] *= 1000  # Extreme values
        
        # Should still produce predictions (though possibly unreliable)
        predictions = predictor.predict(extreme_data[:5])
        assert len(predictions) == 5
        assert not np.isnan(predictions).any()


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system."""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Generate large dataset
        generator = FusionDataGenerator(random_state=42)
        large_data = generator.generate_dataset(n_samples=10000)
        
        start_time = time.time()
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(large_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30.0  # 30 seconds
        assert len(processed_data) <= len(large_data)
        assert X.shape[0] == len(processed_data)
    
    def test_batch_prediction_performance(self):
        """Test performance of batch predictions."""
        # Prepare model
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=1000)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(raw_data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Test batch prediction performance
        batch_size = 1000
        test_data = X[:batch_size]
        
        start_time = time.time()
        predictions = predictor.predict(test_data)
        prediction_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert prediction_time < 5.0  # 5 seconds
        assert len(predictions) == batch_size
        
        # Calculate throughput
        throughput = batch_size / prediction_time
        assert throughput > 100  # At least 100 predictions per second