"""
Fusion prediction models for nuclear fusion analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
except ImportError:
    print("Warning: scikit-learn not available. Model functionality will be limited.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Deep learning models will be disabled.")
    TENSORFLOW_AVAILABLE = False


class FusionPredictor:
    """
    Machine learning predictor for nuclear fusion parameters and performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the fusion predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models."""
        try:
            self.models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0, random_state=42),
                'lasso_regression': Lasso(alpha=1.0, random_state=42),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                ),
                'svr': SVR(kernel='rbf', C=1.0),
                'mlp': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=1000
                )
            }
        except NameError:
            print("Scikit-learn models not available")
            self.models = {}
    
    def create_deep_learning_model(self, input_shape: int, 
                                 output_shape: int = 1) -> 'keras.Model':
        """
        Create a deep learning model for fusion prediction.
        
        Args:
            input_shape: Number of input features
            output_shape: Number of output predictions
            
        Returns:
            Compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for deep learning models")
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(output_shape, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_lstm_model(self, sequence_length: int, 
                         n_features: int,
                         output_shape: int = 1) -> 'keras.Model':
        """
        Create LSTM model for time series fusion prediction.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            output_shape: Number of output predictions
            
        Returns:
            Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM models")
        
        model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, 
                       input_shape=(sequence_length, n_features)),
            layers.Dropout(0.2),
            
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(25),
            layers.Dropout(0.2),
            
            layers.Dense(50, activation='relu'),
            layers.Dense(output_shape, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series, X_val: pd.DataFrame = None,
                   y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Training results dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Make predictions
        train_predictions = model.predict(X_train)
        val_predictions = None
        
        if X_val is not None and y_val is not None:
            val_predictions = model.predict(X_val)
        
        # Calculate metrics
        results = self._calculate_metrics(
            y_train, train_predictions, 
            y_val, val_predictions,
            model_name
        )
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = dict(
                zip(X_train.columns, model.feature_importances_)
            )
        elif hasattr(model, 'coef_'):
            self.feature_importance[model_name] = dict(
                zip(X_train.columns, abs(model.coef_))
            )
        
        self.model_performance[model_name] = results
        
        return results
    
    def train_deep_learning_model(self, X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                X_val: pd.DataFrame = None,
                                y_val: pd.Series = None,
                                epochs: int = 100,
                                batch_size: int = 32) -> Dict[str, Any]:
        """
        Train deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for deep learning")
        
        # Create model
        model = self.create_deep_learning_model(
            input_shape=X_train.shape[1],
            output_shape=1
        )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val.values, y_val.values)
        
        # Train model
        history = model.fit(
            X_train.values, y_train.values,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=20, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=10, factor=0.5
                )
            ]
        )
        
        # Store trained model
        self.trained_models['deep_learning'] = model
        
        # Make predictions
        train_predictions = model.predict(X_train.values).flatten()
        val_predictions = None
        
        if X_val is not None:
            val_predictions = model.predict(X_val.values).flatten()
        
        # Calculate metrics
        results = self._calculate_metrics(
            y_train, train_predictions,
            y_val, val_predictions,
            'deep_learning'
        )
        
        results['training_history'] = history.history
        self.model_performance['deep_learning'] = results
        
        return results
    
    def _calculate_metrics(self, y_train: pd.Series, train_pred: np.ndarray,
                          y_val: pd.Series = None, val_pred: np.ndarray = None,
                          model_name: str = '') -> Dict[str, Any]:
        """
        Calculate model performance metrics.
        
        Args:
            y_train: Training target values
            train_pred: Training predictions
            y_val: Validation target values
            val_pred: Validation predictions
            model_name: Name of the model
            
        Returns:
            Metrics dictionary
        """
        try:
            metrics = {
                'model_name': model_name,
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred)
            }
            
            if y_val is not None and val_pred is not None:
                metrics.update({
                    'val_mse': mean_squared_error(y_val, val_pred),
                    'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                    'val_mae': mean_absolute_error(y_val, val_pred),
                    'val_r2': r2_score(y_val, val_pred)
                })
            
            return metrics
        except NameError:
            # Fallback if sklearn metrics not available
            return {
                'model_name': model_name,
                'train_mse': np.mean((y_train - train_pred) ** 2),
                'train_rmse': np.sqrt(np.mean((y_train - train_pred) ** 2)),
                'train_mae': np.mean(np.abs(y_train - train_pred))
            }
    
    def hyperparameter_optimization(self, model_name: str, 
                                  X_train: pd.DataFrame, 
                                  y_train: pd.Series,
                                  param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for optimization
            
        Returns:
            Optimization results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            if param_grid is None:
                param_grid = self._get_default_param_grid(model_name)
            
            model = self.models[model_name]
            
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        except NameError:
            print("GridSearchCV not available")
            return {}
    
    def _get_default_param_grid(self, model_name: str) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter optimization."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'svr': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def train_all_models(self, data: pd.DataFrame, 
                        target_column: str = 'q_factor',
                        test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            data: Input dataset
            target_column: Target variable name
            test_size: Fraction for test split
            
        Returns:
            Combined training results
        """
        from src.data.processor import FusionDataProcessor
        
        # Preprocess data
        processor = FusionDataProcessor()
        processed_data = processor.preprocess_pipeline(
            data, target_column, test_size
        )
        
        X_train = processed_data['X_train']
        X_val = processed_data['X_val']
        y_train = processed_data['y_train']
        y_val = processed_data['y_val']
        
        results = {}
        
        # Train traditional ML models
        for model_name in self.models.keys():
            try:
                print(f"Training {model_name}...")
                model_results = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                results[model_name] = model_results
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Train deep learning model if available
        if TENSORFLOW_AVAILABLE:
            try:
                print("Training deep learning model...")
                dl_results = self.train_deep_learning_model(
                    X_train, y_train, X_val, y_val
                )
                results['deep_learning'] = dl_results
            except Exception as e:
                print(f"Failed to train deep learning model: {e}")
                results['deep_learning'] = {'error': str(e)}
        
        # Store processed data
        results['processed_data'] = processed_data
        
        return results
    
    def predict(self, X: pd.DataFrame, 
               model_name: str = 'random_forest') -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Input features
            model_name: Name of model to use for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.trained_models[model_name]
        
        if model_name == 'deep_learning' and TENSORFLOW_AVAILABLE:
            return model.predict(X.values).flatten()
        else:
            return model.predict(X)
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance for specified model.
        
        Args:
            model_name: Model name (if None, returns all)
            
        Returns:
            Feature importance dictionary
        """
        if model_name:
            return self.feature_importance.get(model_name, {})
        else:
            return self.feature_importance
    
    def save_models(self, save_path: str = 'saved_models/') -> None:
        """
        Save trained models to disk.
        
        Args:
            save_path: Directory to save models
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            if model_name == 'deep_learning' and TENSORFLOW_AVAILABLE:
                model.save(f"{save_path}/{model_name}.h5")
            else:
                try:
                    joblib.dump(model, f"{save_path}/{model_name}.joblib")
                except NameError:
                    print(f"Cannot save {model_name}: joblib not available")
    
    def load_models(self, load_path: str = 'saved_models/') -> None:
        """
        Load trained models from disk.
        
        Args:
            load_path: Directory to load models from
        """
        import os
        
        for file in os.listdir(load_path):
            if file.endswith('.joblib'):
                model_name = file.replace('.joblib', '')
                try:
                    self.trained_models[model_name] = joblib.load(
                        f"{load_path}/{file}"
                    )
                except NameError:
                    print(f"Cannot load {model_name}: joblib not available")
            elif file.endswith('.h5') and TENSORFLOW_AVAILABLE:
                model_name = file.replace('.h5', '')
                self.trained_models[model_name] = keras.models.load_model(
                    f"{load_path}/{file}"
                )