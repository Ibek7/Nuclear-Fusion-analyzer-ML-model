"""
Unit tests for FusionPredictor class.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.fusion_predictor import FusionPredictor
from src.data.generator import FusionDataGenerator
from src.data.processor import FusionDataProcessor


class TestFusionPredictor:
    """Test cases for FusionPredictor."""
    
    def test_initialization(self):
        """Test FusionPredictor initialization."""
        predictor = FusionPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'models')
        assert hasattr(predictor, 'trained_models')
        assert hasattr(predictor, 'feature_importance')
        
        # Check that models dictionary contains expected models
        expected_models = ['random_forest', 'gradient_boosting', 'support_vector']
        for model in expected_models:
            assert model in predictor.models
    
    def test_prepare_sample_data(self):
        """Test sample data preparation for testing."""
        # Generate sample data for testing
        generator = FusionDataGenerator(random_state=42)
        raw_data = generator.generate_dataset(n_samples=100)
        
        processor = FusionDataProcessor(random_state=42)
        processed_data = processor.preprocess_pipeline(
            raw_data, 
            target_column='q_factor',
            test_size=0.2
        )
        
        return processed_data
    
    def test_train_model_random_forest(self):
        """Test training random forest model."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train random forest
        results = predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        
        # Check that training completed
        assert 'random_forest' in predictor.trained_models
        assert results is not None
        
        # Check that results contain expected metrics
        expected_keys = ['train_r2', 'val_r2', 'train_rmse', 'val_rmse', 'train_mae', 'val_mae']
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"
        
        # Check that metrics are reasonable
        assert 0 <= results['train_r2'] <= 1, "Training R² should be between 0 and 1"
        assert 0 <= results['val_r2'] <= 1, "Validation R² should be between 0 and 1"
        assert results['train_rmse'] >= 0, "RMSE should be non-negative"
        assert results['val_rmse'] >= 0, "RMSE should be non-negative"
        
        # Check feature importance
        assert 'random_forest' in predictor.feature_importance
        importance = predictor.feature_importance['random_forest']
        assert len(importance) == X_train.shape[1], "Should have importance for each feature"
        assert all(imp >= 0 for imp in importance.values()), "Importance should be non-negative"
    
    def test_train_model_gradient_boosting(self):
        """Test training gradient boosting model."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train gradient boosting
        results = predictor.train_model('gradient_boosting', X_train, y_train, X_val, y_val)
        
        # Check training completed
        assert 'gradient_boosting' in predictor.trained_models
        assert results is not None
        
        # Check metrics
        assert 'train_r2' in results
        assert 'val_r2' in results
        assert results['train_r2'] >= 0
        assert results['val_r2'] >= 0
    
    def test_train_model_support_vector(self):
        """Test training support vector regression model."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train SVR
        results = predictor.train_model('support_vector', X_train, y_train, X_val, y_val)
        
        # Check training completed
        assert 'support_vector' in predictor.trained_models
        assert results is not None
        
        # SVR doesn't have feature_importances, so check this is handled
        # (Should not raise an error)
    
    def test_train_invalid_model(self):
        """Test training with invalid model name."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Should raise error for invalid model
        with pytest.raises(ValueError):
            predictor.train_model('invalid_model', X_train, y_train, X_val, y_val)
    
    @patch('src.models.fusion_predictor.tf')
    def test_train_deep_learning_model_mock(self, mock_tf):
        """Test deep learning model training with mocked TensorFlow."""
        # Mock TensorFlow components
        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'loss': [1.0, 0.5], 'val_loss': [1.1, 0.6]}
        mock_model.predict.return_value = np.array([[1.0], [1.5], [0.8]])
        
        mock_tf.keras.Sequential.return_value = mock_model
        mock_tf.keras.layers = MagicMock()
        mock_tf.keras.optimizers = MagicMock()
        
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train deep learning model
        results = predictor.train_deep_learning_model(
            X_train, y_train, X_val, y_val, epochs=2
        )
        
        # Check that training completed
        assert results is not None
        assert 'train_r2' in results
        assert 'val_r2' in results
        assert 'deep_learning' in predictor.trained_models
    
    def test_predict_single_model(self):
        """Test prediction with single model."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        
        # Train a model first
        predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = predictor.predict(X_test, 'random_forest')
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
        assert all(np.isfinite(predictions)), "Predictions should be finite"
    
    def test_predict_ensemble(self):
        """Test ensemble prediction with multiple models."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        
        # Train multiple models
        predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        predictor.train_model('gradient_boosting', X_train, y_train, X_val, y_val)
        
        # Make ensemble predictions
        predictions = predictor.predict_ensemble(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
        assert all(np.isfinite(predictions)), "Ensemble predictions should be finite"
    
    def test_predict_untrained_model(self):
        """Test prediction with untrained model."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        X_test = processed_data['X_test']
        
        # Should raise error for untrained model
        with pytest.raises(ValueError):
            predictor.predict(X_test, 'random_forest')
    
    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Test with small parameter grid and few iterations for speed
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        best_params, best_score = predictor.optimize_hyperparameters(
            'random_forest', X_train, y_train, param_grid, cv=2
        )
        
        # Check that optimization completed
        assert best_params is not None
        assert best_score is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        
        # Check that best parameters are from the grid
        for param, value in best_params.items():
            if param in param_grid:
                assert value in param_grid[param]
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train a model
        predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        
        # Get feature importance
        importance = predictor.get_feature_importance('random_forest')
        
        # Check importance
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == X_train.shape[1]
        assert all(isinstance(imp, (int, float)) for imp in importance.values())
        assert all(imp >= 0 for imp in importance.values())
    
    def test_save_and_load_models(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        
        # Train a model
        predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        original_predictions = predictor.predict(X_test, 'random_forest')
        
        # Save models to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            predictor.save_models(tmp_dir)
            
            # Check that files were created
            saved_files = os.listdir(tmp_dir)
            assert any(f.startswith('random_forest') and f.endswith('.joblib') for f in saved_files)
            
            # Create new predictor and load models
            new_predictor = FusionPredictor()
            new_predictor.load_models(tmp_dir)
            
            # Check that model was loaded
            assert 'random_forest' in new_predictor.trained_models
            
            # Check that predictions are the same
            loaded_predictions = new_predictor.predict(X_test, 'random_forest')
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
    
    def test_input_validation(self):
        """Test input validation."""
        predictor = FusionPredictor()
        
        # Test with invalid input shapes
        X_wrong_shape = np.array([[1, 2], [3, 4]])  # 2D but might be wrong features
        y_wrong_shape = np.array([1, 2, 3])  # Different length
        
        with pytest.raises((ValueError, TypeError)):
            predictor.train_model('random_forest', X_wrong_shape, y_wrong_shape, X_wrong_shape, y_wrong_shape)
    
    def test_model_performance_bounds(self):
        """Test that model performance is within reasonable bounds."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Train model
        results = predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        
        # Check that performance is reasonable (not perfect, not terrible)
        # For synthetic data with good signal, we expect decent performance
        assert results['train_r2'] > 0.1, "Training R² should be reasonably good"
        assert results['val_r2'] > 0.05, "Validation R² should be positive"
        
        # Training should generally be better than validation (some overfitting expected)
        assert results['train_r2'] >= results['val_r2'] - 0.1, "Training should not be much worse than validation"
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent across multiple calls."""
        predictor = FusionPredictor()
        processed_data = self.test_prepare_sample_data()
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        
        # Train model
        predictor.train_model('random_forest', X_train, y_train, X_val, y_val)
        
        # Make predictions multiple times
        pred1 = predictor.predict(X_test, 'random_forest')
        pred2 = predictor.predict(X_test, 'random_forest')
        
        # Should be identical
        np.testing.assert_array_equal(pred1, pred2)