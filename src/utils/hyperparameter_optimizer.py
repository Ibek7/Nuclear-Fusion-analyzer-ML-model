"""
Hyperparameter optimization system for nuclear fusion ML models.

This module provides advanced hyperparameter tuning capabilities including
grid search, random search, Bayesian optimization, and multi-objective
optimization for fusion prediction models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Import optimization libraries with fallbacks
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available, some optimization features disabled")

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn("scikit-optimize not available, Bayesian optimization disabled")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available, advanced optimization features disabled")

from src.models.fusion_predictor import FusionPredictor
from src.utils.evaluator import FusionModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for hyperparameter optimization results."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    optimization_time: float = 0.0
    method: str = ""
    objective_name: str = ""
    cv_results: Optional[Dict[str, Any]] = None


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    
    param_name: str
    param_type: str  # 'real', 'integer', 'categorical'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    description: str = ""


class FusionHyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for fusion ML models.
    
    Supports multiple optimization strategies including grid search,
    random search, Bayesian optimization, and multi-objective optimization.
    """
    
    # Predefined parameter spaces for common fusion models
    PARAMETER_SPACES = {
        'random_forest': {
            'n_estimators': HyperparameterSpace(
                'n_estimators', 'integer', (50, 500), 
                description='Number of trees in the forest'
            ),
            'max_depth': HyperparameterSpace(
                'max_depth', 'integer', (3, 20),
                description='Maximum depth of trees'
            ),
            'min_samples_split': HyperparameterSpace(
                'min_samples_split', 'integer', (2, 20),
                description='Minimum samples required to split a node'
            ),
            'min_samples_leaf': HyperparameterSpace(
                'min_samples_leaf', 'integer', (1, 10),
                description='Minimum samples required at a leaf node'
            ),
            'max_features': HyperparameterSpace(
                'max_features', 'categorical', choices=['sqrt', 'log2', None],
                description='Number of features to consider for best split'
            )
        },
        'gradient_boosting': {
            'n_estimators': HyperparameterSpace(
                'n_estimators', 'integer', (50, 300),
                description='Number of boosting stages'
            ),
            'learning_rate': HyperparameterSpace(
                'learning_rate', 'real', (0.01, 0.3), log_scale=True,
                description='Learning rate shrinks contribution of each tree'
            ),
            'max_depth': HyperparameterSpace(
                'max_depth', 'integer', (3, 15),
                description='Maximum depth of individual regression estimators'
            ),
            'subsample': HyperparameterSpace(
                'subsample', 'real', (0.6, 1.0),
                description='Fraction of samples used for fitting trees'
            ),
            'min_samples_split': HyperparameterSpace(
                'min_samples_split', 'integer', (2, 20),
                description='Minimum samples required to split a node'
            )
        },
        'neural_network': {
            'hidden_layer_sizes': HyperparameterSpace(
                'hidden_layer_sizes', 'categorical',
                choices=[(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100)],
                description='Number of neurons in hidden layers'
            ),
            'learning_rate_init': HyperparameterSpace(
                'learning_rate_init', 'real', (0.0001, 0.1), log_scale=True,
                description='Initial learning rate'
            ),
            'alpha': HyperparameterSpace(
                'alpha', 'real', (0.0001, 0.1), log_scale=True,
                description='L2 penalty parameter'
            ),
            'batch_size': HyperparameterSpace(
                'batch_size', 'categorical', choices=[32, 64, 128, 256],
                description='Size of minibatches for stochastic optimizers'
            ),
            'max_iter': HyperparameterSpace(
                'max_iter', 'integer', (200, 1000),
                description='Maximum number of iterations'
            )
        },
        'svr': {
            'C': HyperparameterSpace(
                'C', 'real', (0.1, 100), log_scale=True,
                description='Regularization parameter'
            ),
            'epsilon': HyperparameterSpace(
                'epsilon', 'real', (0.01, 1.0), log_scale=True,
                description='Epsilon in epsilon-SVR model'
            ),
            'gamma': HyperparameterSpace(
                'gamma', 'categorical', choices=['scale', 'auto'],
                description='Kernel coefficient'
            ),
            'kernel': HyperparameterSpace(
                'kernel', 'categorical', choices=['rbf', 'poly', 'sigmoid'],
                description='Kernel type'
            )
        }
    }
    
    def __init__(
        self, 
        predictor: FusionPredictor,
        evaluator: Optional[FusionModelEvaluator] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            predictor: FusionPredictor instance for model training.
            evaluator: Model evaluator for scoring (creates new if None).
            n_jobs: Number of parallel jobs (-1 for all cores).
            random_state: Random state for reproducibility.
        """
        self.predictor = predictor
        self.evaluator = evaluator or FusionModelEvaluator()
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Optimization history
        self.optimization_results: Dict[str, List[OptimizationResult]] = {}
        
        logger.info(f"FusionHyperparameterOptimizer initialized with n_jobs={n_jobs}")
    
    def grid_search_optimization(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
        scoring: str = 'r2'
    ) -> OptimizationResult:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            param_grid: Parameter grid (uses default if None).
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for optimization.
            
        Returns:
            OptimizationResult: Results of the optimization.
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for grid search optimization")
        
        start_time = time.time()
        
        # Get parameter grid
        if param_grid is None:
            param_grid = self._get_sklearn_param_grid(model_type)
        
        # Get base model
        base_model = self.predictor._get_base_model(model_type)
        
        # Set up grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            return_train_score=True,
            verbose=1
        )
        
        # Fit grid search
        logger.info(f"Starting grid search for {model_type} with {len(ParameterGrid(param_grid))} combinations")
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model on validation set
        val_score = grid_search.score(X_val, y_val)
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        result = OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=val_score,
            best_model=grid_search.best_estimator_,
            total_evaluations=len(grid_search.cv_results_['mean_test_score']),
            optimization_time=optimization_time,
            method='grid_search',
            objective_name=scoring,
            cv_results=grid_search.cv_results_
        )
        
        # Store result
        if model_type not in self.optimization_results:
            self.optimization_results[model_type] = []
        self.optimization_results[model_type].append(result)
        
        logger.info(
            f"Grid search completed for {model_type}: "
            f"best_score={val_score:.4f}, "
            f"evaluations={result.total_evaluations}, "
            f"time={optimization_time:.1f}s"
        )
        
        return result
    
    def random_search_optimization(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_distributions: Optional[Dict[str, Any]] = None,
        n_iter: int = 100,
        cv_folds: int = 5,
        scoring: str = 'r2'
    ) -> OptimizationResult:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            param_distributions: Parameter distributions (uses default if None).
            n_iter: Number of parameter settings sampled.
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for optimization.
            
        Returns:
            OptimizationResult: Results of the optimization.
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for random search optimization")
        
        start_time = time.time()
        
        # Get parameter distributions
        if param_distributions is None:
            param_distributions = self._get_sklearn_param_distributions(model_type)
        
        # Get base model
        base_model = self.predictor._get_base_model(model_type)
        
        # Set up random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            return_train_score=True,
            random_state=self.random_state,
            verbose=1
        )
        
        # Fit random search
        logger.info(f"Starting random search for {model_type} with {n_iter} iterations")
        random_search.fit(X_train, y_train)
        
        # Evaluate best model on validation set
        val_score = random_search.score(X_val, y_val)
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        result = OptimizationResult(
            best_params=random_search.best_params_,
            best_score=val_score,
            best_model=random_search.best_estimator_,
            total_evaluations=n_iter,
            optimization_time=optimization_time,
            method='random_search',
            objective_name=scoring,
            cv_results=random_search.cv_results_
        )
        
        # Store result
        if model_type not in self.optimization_results:
            self.optimization_results[model_type] = []
        self.optimization_results[model_type].append(result)
        
        logger.info(
            f"Random search completed for {model_type}: "
            f"best_score={val_score:.4f}, "
            f"evaluations={result.total_evaluations}, "
            f"time={optimization_time:.1f}s"
        )
        
        return result
    
    def bayesian_optimization(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        space: Optional[List] = None,
        n_calls: int = 50,
        method: str = 'gp'
    ) -> OptimizationResult:
        """
        Perform Bayesian hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            space: Search space (uses default if None).
            n_calls: Number of function evaluations.
            method: Optimization method ('gp', 'forest', 'gbrt').
            
        Returns:
            OptimizationResult: Results of the optimization.
        """
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        start_time = time.time()
        
        # Get search space
        if space is None:
            space = self._get_skopt_search_space(model_type)
        
        # Get parameter names
        param_names = [dim.name for dim in space]
        
        # Define objective function
        @use_named_args(space)
        def objective(**params):
            try:
                # Train model with current parameters
                model = self.predictor._get_base_model(model_type)
                model.set_params(**params)
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                score = model.score(X_val, y_val)
                
                # Return negative score for minimization
                return -score
                
            except Exception as e:
                logger.warning(f"Objective function failed with params {params}: {e}")
                return 1000  # Large positive value for minimization
        
        # Choose optimization method
        if method == 'gp':
            optimizer_func = gp_minimize
        elif method == 'forest':
            optimizer_func = forest_minimize
        elif method == 'gbrt':
            optimizer_func = gbrt_minimize
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        logger.info(f"Starting Bayesian optimization ({method}) for {model_type} with {n_calls} calls")
        
        # Run optimization
        result_skopt = optimizer_func(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=self.random_state,
            verbose=True
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result_skopt.x))
        best_score = -result_skopt.fun
        
        # Train final model with best parameters
        best_model = self.predictor._get_base_model(model_type)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            total_evaluations=n_calls,
            optimization_time=optimization_time,
            method=f'bayesian_{method}',
            objective_name='r2'
        )
        
        # Store optimization history
        result.optimization_history = [
            {'iteration': i, 'score': -score, 'params': dict(zip(param_names, x))}
            for i, (x, score) in enumerate(zip(result_skopt.x_iters, result_skopt.func_vals))
        ]
        
        # Store result
        if model_type not in self.optimization_results:
            self.optimization_results[model_type] = []
        self.optimization_results[model_type].append(result)
        
        logger.info(
            f"Bayesian optimization completed for {model_type}: "
            f"best_score={best_score:.4f}, "
            f"evaluations={result.total_evaluations}, "
            f"time={optimization_time:.1f}s"
        )
        
        return result
    
    def optuna_optimization(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        study_name: Optional[str] = None
    ) -> OptimizationResult:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            model_type: Type of model to optimize.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            n_trials: Number of optimization trials.
            study_name: Name of the Optuna study.
            
        Returns:
            OptimizationResult: Results of the optimization.
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for advanced optimization")
        
        start_time = time.time()
        
        if study_name is None:
            study_name = f"{model_type}_optimization"
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Define objective function
        def objective(trial):
            try:
                # Sample hyperparameters
                params = self._sample_optuna_params(trial, model_type)
                
                # Train model
                model = self.predictor._get_base_model(model_type)
                model.set_params(**params)
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                score = model.score(X_val, y_val)
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed with params {trial.params}: {e}")
                return -1000  # Large negative value for maximization
        
        logger.info(f"Starting Optuna optimization for {model_type} with {n_trials} trials")
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        
        # Train final model with best parameters
        best_model = self.predictor._get_base_model(model_type)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            total_evaluations=len(study.trials),
            optimization_time=optimization_time,
            method='optuna_tpe',
            objective_name='r2'
        )
        
        # Store optimization history
        result.optimization_history = [
            {
                'iteration': i,
                'score': trial.value if trial.value is not None else -1000,
                'params': trial.params
            }
            for i, trial in enumerate(study.trials)
        ]
        
        # Store result
        if model_type not in self.optimization_results:
            self.optimization_results[model_type] = []
        self.optimization_results[model_type].append(result)
        
        logger.info(
            f"Optuna optimization completed for {model_type}: "
            f"best_score={best_score:.4f}, "
            f"evaluations={result.total_evaluations}, "
            f"time={optimization_time:.1f}s"
        )
        
        return result
    
    def compare_optimization_methods(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        methods: List[str] = ['grid_search', 'random_search', 'bayesian_gp'],
        n_evaluations: int = 50
    ) -> Dict[str, OptimizationResult]:
        """
        Compare different optimization methods for a model.
        
        Args:
            model_type: Type of model to optimize.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            methods: List of optimization methods to compare.
            n_evaluations: Number of evaluations for each method.
            
        Returns:
            Dictionary mapping method names to optimization results.
        """
        results = {}
        
        for method in methods:
            logger.info(f"Running optimization method: {method}")
            
            try:
                if method == 'grid_search':
                    result = self.grid_search_optimization(
                        model_type, X_train, y_train, X_val, y_val
                    )
                elif method == 'random_search':
                    result = self.random_search_optimization(
                        model_type, X_train, y_train, X_val, y_val,
                        n_iter=n_evaluations
                    )
                elif method.startswith('bayesian_'):
                    bayes_method = method.split('_')[1]
                    result = self.bayesian_optimization(
                        model_type, X_train, y_train, X_val, y_val,
                        n_calls=n_evaluations, method=bayes_method
                    )
                elif method == 'optuna':
                    result = self.optuna_optimization(
                        model_type, X_train, y_train, X_val, y_val,
                        n_trials=n_evaluations
                    )
                else:
                    logger.warning(f"Unknown optimization method: {method}")
                    continue
                
                results[method] = result
                
            except Exception as e:
                logger.error(f"Optimization method {method} failed: {e}")
                continue
        
        return results
    
    def _get_sklearn_param_grid(self, model_type: str) -> Dict[str, List]:
        """Get sklearn parameter grid for grid search."""
        if model_type not in self.PARAMETER_SPACES:
            raise ValueError(f"No parameter space defined for {model_type}")
        
        param_grid = {}
        space = self.PARAMETER_SPACES[model_type]
        
        for param_name, param_space in space.items():
            if param_space.param_type == 'categorical':
                param_grid[param_name] = param_space.choices
            elif param_space.param_type == 'integer':
                min_val, max_val = param_space.bounds
                param_grid[param_name] = list(range(int(min_val), int(max_val) + 1, 
                                                   max(1, (int(max_val) - int(min_val)) // 10)))
            elif param_space.param_type == 'real':
                min_val, max_val = param_space.bounds
                if param_space.log_scale:
                    param_grid[param_name] = np.logspace(
                        np.log10(min_val), np.log10(max_val), 5
                    ).tolist()
                else:
                    param_grid[param_name] = np.linspace(min_val, max_val, 5).tolist()
        
        return param_grid
    
    def _get_sklearn_param_distributions(self, model_type: str) -> Dict[str, Any]:
        """Get sklearn parameter distributions for random search."""
        from scipy.stats import randint, uniform, loguniform
        
        if model_type not in self.PARAMETER_SPACES:
            raise ValueError(f"No parameter space defined for {model_type}")
        
        param_distributions = {}
        space = self.PARAMETER_SPACES[model_type]
        
        for param_name, param_space in space.items():
            if param_space.param_type == 'categorical':
                param_distributions[param_name] = param_space.choices
            elif param_space.param_type == 'integer':
                min_val, max_val = param_space.bounds
                param_distributions[param_name] = randint(int(min_val), int(max_val) + 1)
            elif param_space.param_type == 'real':
                min_val, max_val = param_space.bounds
                if param_space.log_scale:
                    param_distributions[param_name] = loguniform(min_val, max_val)
                else:
                    param_distributions[param_name] = uniform(min_val, max_val - min_val)
        
        return param_distributions
    
    def _get_skopt_search_space(self, model_type: str) -> List:
        """Get scikit-optimize search space."""
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        if model_type not in self.PARAMETER_SPACES:
            raise ValueError(f"No parameter space defined for {model_type}")
        
        space = []
        param_space = self.PARAMETER_SPACES[model_type]
        
        for param_name, param_def in param_space.items():
            if param_def.param_type == 'categorical':
                space.append(Categorical(param_def.choices, name=param_name))
            elif param_def.param_type == 'integer':
                min_val, max_val = param_def.bounds
                space.append(Integer(int(min_val), int(max_val), name=param_name))
            elif param_def.param_type == 'real':
                min_val, max_val = param_def.bounds
                if param_def.log_scale:
                    space.append(Real(min_val, max_val, prior='log-uniform', name=param_name))
                else:
                    space.append(Real(min_val, max_val, name=param_name))
        
        return space
    
    def _sample_optuna_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        if model_type not in self.PARAMETER_SPACES:
            raise ValueError(f"No parameter space defined for {model_type}")
        
        params = {}
        param_space = self.PARAMETER_SPACES[model_type]
        
        for param_name, param_def in param_space.items():
            if param_def.param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_def.choices)
            elif param_def.param_type == 'integer':
                min_val, max_val = param_def.bounds
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            elif param_def.param_type == 'real':
                min_val, max_val = param_def.bounds
                if param_def.log_scale:
                    params[param_name] = trial.suggest_loguniform(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_uniform(param_name, min_val, max_val)
        
        return params
    
    def export_optimization_results(self, filepath: str):
        """Export optimization results to JSON file."""
        export_data = {}
        
        for model_type, results in self.optimization_results.items():
            export_data[model_type] = []
            
            for result in results:
                result_dict = {
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'total_evaluations': result.total_evaluations,
                    'optimization_time': result.optimization_time,
                    'method': result.method,
                    'objective_name': result.objective_name,
                    'optimization_history': result.optimization_history
                }
                export_data[model_type].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Optimization results exported to {filepath}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        summary = {}
        
        for model_type, results in self.optimization_results.items():
            if not results:
                continue
            
            best_result = max(results, key=lambda x: x.best_score)
            
            method_comparison = {}
            for result in results:
                method_comparison[result.method] = {
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'total_evaluations': result.total_evaluations
                }
            
            summary[model_type] = {
                'best_overall_score': best_result.best_score,
                'best_method': best_result.method,
                'best_params': best_result.best_params,
                'total_optimizations': len(results),
                'method_comparison': method_comparison
            }
        
        return summary