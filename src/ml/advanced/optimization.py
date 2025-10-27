"""
Hyperparameter optimization and model tuning utilities.

This module provides advanced hyperparameter optimization using:
- Bayesian optimization (Optuna)
- Grid search and random search
- Multi-objective optimization
- Automated feature selection
- Neural architecture search
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import warnings

# Core optimization libraries
try:
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, cross_val_score,
        ParameterGrid, ParameterSampler
    )
    from sklearn.metrics import make_scorer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Advanced optimizers
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# Feature selection
try:
    from sklearn.feature_selection import (
        SelectKBest, SelectFromModel, RFE, RFECV,
        mutual_info_regression, f_regression
    )
    HAS_FEATURE_SELECTION = True
except ImportError:
    HAS_FEATURE_SELECTION = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration."""
    
    # General settings
    n_trials: int = 100
    timeout_seconds: Optional[int] = 3600  # 1 hour
    n_jobs: int = -1
    random_state: int = 42
    
    # Optimization method
    method: str = "bayesian"  # bayesian, grid, random, evolution
    
    # Cross-validation
    cv_folds: int = 5
    scoring_metric: str = "r2"
    
    # Bayesian optimization
    sampler: str = "tpe"  # tpe, cmaes
    pruner: str = "median"  # median, hyperband
    
    # Multi-objective
    enable_multi_objective: bool = False
    objectives: List[str] = None
    
    # Feature selection
    enable_feature_selection: bool = True
    max_features_ratio: float = 0.8
    
    # Early stopping
    early_stopping_rounds: int = 20
    min_improvement: float = 0.001


@dataclass
class OptimizationResult:
    """Optimization result container."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    optimization_history: List[Dict[str, Any]]
    feature_importance: Optional[np.ndarray]
    selected_features: Optional[List[int]]
    total_time: float
    n_trials_completed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'selected_features': self.selected_features,
            'total_time': self.total_time,
            'n_trials_completed': self.n_trials_completed
        }


class ParameterSpace:
    """Parameter space definition for optimization."""
    
    def __init__(self):
        """Initialize parameter space."""
        self.parameters = {}
        
    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """Add float parameter."""
        self.parameters[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log': log
        }
        
    def add_int(self, name: str, low: int, high: int, step: int = 1):
        """Add integer parameter."""
        self.parameters[name] = {
            'type': 'int',
            'low': low,
            'high': high,
            'step': step
        }
        
    def add_categorical(self, name: str, choices: List[Any]):
        """Add categorical parameter."""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }
    
    def to_optuna_suggest(self, trial) -> Dict[str, Any]:
        """Convert to Optuna trial suggestions."""
        params = {}
        
        for name, config in self.parameters.items():
            if config['type'] == 'float':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high'],
                    log=config.get('log', False)
                )
            elif config['type'] == 'int':
                params[name] = trial.suggest_int(
                    name, config['low'], config['high'],
                    step=config.get('step', 1)
                )
            elif config['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, config['choices'])
        
        return params
    
    def to_sklearn_space(self) -> Dict[str, List[Any]]:
        """Convert to sklearn parameter space."""
        space = {}
        
        for name, config in self.parameters.items():
            if config['type'] == 'float':
                # Create range for grid search
                if config.get('log', False):
                    space[name] = np.logspace(
                        np.log10(config['low']),
                        np.log10(config['high']),
                        20
                    )
                else:
                    space[name] = np.linspace(config['low'], config['high'], 20)
            elif config['type'] == 'int':
                space[name] = list(range(
                    config['low'], 
                    config['high'] + 1, 
                    config.get('step', 1)
                ))
            elif config['type'] == 'categorical':
                space[name] = config['choices']
        
        return space


class BayesianOptimizer:
    """
    Bayesian optimization using Optuna.
    
    Provides efficient hyperparameter optimization.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: Optimization configuration.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna not available for Bayesian optimization")
        
        self.config = config
        self.study = None
        
        logger.info("BayesianOptimizer initialized")
    
    def optimize(
        self,
        objective_func: Callable,
        param_space: ParameterSpace,
        direction: str = "maximize"
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Objective function to optimize.
            param_space: Parameter space definition.
            direction: Optimization direction ("maximize" or "minimize").
            
        Returns:
            Optimization result.
        """
        start_time = time.time()
        
        # Create sampler
        if self.config.sampler == "tpe":
            sampler = TPESampler(seed=self.config.random_state)
        elif self.config.sampler == "cmaes":
            sampler = CmaEsSampler(seed=self.config.random_state)
        else:
            sampler = TPESampler(seed=self.config.random_state)
        
        # Create pruner
        if self.config.pruner == "median":
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif self.config.pruner == "hyperband":
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=self.config.cv_folds,
                reduction_factor=3
            )
        else:
            pruner = MedianPruner()
        
        # Create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective wrapper
        def optuna_objective(trial):
            params = param_space.to_optuna_suggest(trial)
            return objective_func(params, trial)
        
        # Run optimization
        self.study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=1  # Optuna handles parallelism internally
        )
        
        # Prepare results
        total_time = time.time() - start_time
        
        optimization_history = []
        for trial in self.study.trials:
            optimization_history.append({
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            })
        
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            best_model=None,  # Set by caller
            optimization_history=optimization_history,
            feature_importance=None,
            selected_features=None,
            total_time=total_time,
            n_trials_completed=len(self.study.trials)
        )
        
        logger.info(f"Bayesian optimization completed in {total_time:.2f} seconds")
        logger.info(f"Best score: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")
        
        return result


class GridSearchOptimizer:
    """
    Grid search optimization.
    
    Exhaustive search over parameter combinations.
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize grid search optimizer."""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn not available for grid search")
        
        self.config = config
        
        logger.info("GridSearchOptimizer initialized")
    
    def optimize(
        self,
        model,
        param_space: ParameterSpace,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            model: Model to optimize.
            param_space: Parameter space.
            X: Training features.
            y: Training targets.
            scoring: Scoring metric.
            
        Returns:
            Optimization result.
        """
        start_time = time.time()
        
        # Convert parameter space
        param_grid = param_space.to_sklearn_space()
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            scoring=scoring or self.config.scoring_metric,
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Prepare results
        total_time = time.time() - start_time
        
        optimization_history = []
        for i, (params, score) in enumerate(zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score']
        )):
            optimization_history.append({
                'trial_number': i,
                'params': params,
                'value': score,
                'state': 'COMPLETE'
            })
        
        result = OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            best_model=grid_search.best_estimator_,
            optimization_history=optimization_history,
            feature_importance=None,
            selected_features=None,
            total_time=total_time,
            n_trials_completed=len(grid_search.cv_results_['params'])
        )
        
        logger.info(f"Grid search completed in {total_time:.2f} seconds")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return result


class FeatureSelector:
    """
    Advanced feature selection utilities.
    
    Provides multiple feature selection methods.
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize feature selector."""
        if not HAS_FEATURE_SELECTION:
            raise RuntimeError("Feature selection libraries not available")
        
        self.config = config
        self.selector = None
        self.selected_features_ = None
        self.feature_scores_ = None
        
        logger.info("FeatureSelector initialized")
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "mutual_info"
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Select features using specified method.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            method: Feature selection method.
            
        Returns:
            Tuple of (selected_X, selected_feature_indices).
        """
        n_features = min(
            int(X.shape[1] * self.config.max_features_ratio),
            X.shape[1]
        )
        
        if method == "mutual_info":
            self.selector = SelectKBest(
                score_func=mutual_info_regression,
                k=n_features
            )
        elif method == "f_regression":
            self.selector = SelectKBest(
                score_func=f_regression,
                k=n_features
            )
        elif method == "rfe":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.config.random_state
            )
            self.selector = RFE(estimator, n_features_to_select=n_features)
        elif method == "model_based":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.config.random_state
            )
            estimator.fit(X, y)
            self.selector = SelectFromModel(estimator, max_features=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit selector and transform data
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected feature indices
        if hasattr(self.selector, 'get_support'):
            self.selected_features_ = np.where(self.selector.get_support())[0].tolist()
        else:
            self.selected_features_ = list(range(X_selected.shape[1]))
        
        # Get feature scores if available
        if hasattr(self.selector, 'scores_'):
            self.feature_scores_ = self.selector.scores_
        elif hasattr(self.selector, 'ranking_'):
            self.feature_scores_ = 1.0 / self.selector.ranking_
        
        logger.info(f"Selected {len(self.selected_features_)} features using {method}")
        
        return X_selected, self.selected_features_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted selector."""
        if self.selector is None:
            raise ValueError("Feature selector not fitted")
        
        return self.selector.transform(X)


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization framework.
    
    Supports multiple optimization methods and objectives.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Optimization configuration.
        """
        self.config = config
        self.feature_selector = None
        if config.enable_feature_selection:
            self.feature_selector = FeatureSelector(config)
        
        logger.info("HyperparameterOptimizer initialized")
    
    def optimize_model(
        self,
        model_class,
        param_space: ParameterSpace,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Optimize model hyperparameters.
        
        Args:
            model_class: Model class to optimize.
            param_space: Parameter space definition.
            X: Training features.
            y: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            
        Returns:
            Optimization result.
        """
        # Feature selection
        selected_features = None
        if self.config.enable_feature_selection:
            X, selected_features = self.feature_selector.select_features(X, y)
            if X_val is not None:
                X_val = self.feature_selector.transform(X_val)
        
        # Define objective function
        def objective_func(params, trial=None):
            try:
                # Create model with parameters
                model = model_class(**params)
                
                # Cross-validation score
                if X_val is not None and y_val is not None:
                    # Use validation set
                    model.fit(X, y)
                    predictions = model.predict(X_val)
                    
                    if self.config.scoring_metric == "r2":
                        from sklearn.metrics import r2_score
                        score = r2_score(y_val, predictions)
                    elif self.config.scoring_metric == "neg_mse":
                        from sklearn.metrics import mean_squared_error
                        score = -mean_squared_error(y_val, predictions)
                    else:
                        score = 0.0
                else:
                    # Use cross-validation
                    scores = cross_val_score(
                        model, X, y,
                        cv=self.config.cv_folds,
                        scoring=self.config.scoring_metric,
                        n_jobs=1  # Avoid nested parallelism
                    )
                    score = scores.mean()
                
                # Report intermediate value for pruning
                if trial is not None:
                    trial.report(score, step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.config.scoring_metric == "r2" else float('inf')
        
        # Choose optimization method
        if self.config.method == "bayesian" and HAS_OPTUNA:
            optimizer = BayesianOptimizer(self.config)
            result = optimizer.optimize(objective_func, param_space)
        elif self.config.method == "grid":
            model_instance = model_class()
            optimizer = GridSearchOptimizer(self.config)
            result = optimizer.optimize(model_instance, param_space, X, y)
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.method}")
        
        # Train best model
        best_model = model_class(**result.best_params)
        best_model.fit(X, y)
        result.best_model = best_model
        
        # Add feature selection results
        if selected_features is not None:
            result.selected_features = selected_features
            if hasattr(self.feature_selector, 'feature_scores_'):
                result.feature_importance = self.feature_selector.feature_scores_
        
        return result
    
    def multi_objective_optimize(
        self,
        model_class,
        param_space: ParameterSpace,
        X: np.ndarray,
        y: np.ndarray,
        objectives: List[str]
    ) -> List[OptimizationResult]:
        """
        Multi-objective optimization.
        
        Args:
            model_class: Model class to optimize.
            param_space: Parameter space.
            X: Training features.
            y: Training targets.
            objectives: List of objectives to optimize.
            
        Returns:
            Pareto-optimal solutions.
        """
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna required for multi-objective optimization")
        
        # Define multi-objective function
        def multi_objective_func(trial):
            params = param_space.to_optuna_suggest(trial)
            model = model_class(**params)
            
            # Calculate objectives
            objective_values = []
            
            for objective in objectives:
                if objective == "accuracy":
                    scores = cross_val_score(model, X, y, cv=self.config.cv_folds, scoring="r2")
                    objective_values.append(scores.mean())
                elif objective == "speed":
                    # Measure training time
                    start_time = time.time()
                    model.fit(X, y)
                    training_time = time.time() - start_time
                    objective_values.append(-training_time)  # Minimize time
                elif objective == "complexity":
                    # Model complexity (e.g., number of parameters)
                    if hasattr(model, 'n_estimators'):
                        complexity = -model.n_estimators  # Minimize complexity
                    else:
                        complexity = -len(params)
                    objective_values.append(complexity)
            
            return objective_values
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=["maximize"] * len(objectives),
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(multi_objective_func, n_trials=self.config.n_trials)
        
        # Return Pareto-optimal solutions
        results = []
        for trial in study.best_trials:
            result = OptimizationResult(
                best_params=trial.params,
                best_score=trial.values[0],  # First objective as primary
                best_model=model_class(**trial.params),
                optimization_history=[],
                feature_importance=None,
                selected_features=None,
                total_time=0.0,
                n_trials_completed=1
            )
            results.append(result)
        
        return results


def create_model_parameter_spaces() -> Dict[str, ParameterSpace]:
    """
    Create predefined parameter spaces for common models.
    
    Returns:
        Dictionary of parameter spaces.
    """
    spaces = {}
    
    # Random Forest
    rf_space = ParameterSpace()
    rf_space.add_int("n_estimators", 50, 500)
    rf_space.add_int("max_depth", 3, 20)
    rf_space.add_float("min_samples_split", 0.01, 0.3)
    rf_space.add_float("min_samples_leaf", 0.01, 0.2)
    rf_space.add_categorical("max_features", ["sqrt", "log2", None])
    spaces["random_forest"] = rf_space
    
    # XGBoost
    if HAS_XGB:
        xgb_space = ParameterSpace()
        xgb_space.add_int("n_estimators", 50, 500)
        xgb_space.add_int("max_depth", 3, 15)
        xgb_space.add_float("learning_rate", 0.01, 0.3, log=True)
        xgb_space.add_float("subsample", 0.6, 1.0)
        xgb_space.add_float("colsample_bytree", 0.6, 1.0)
        xgb_space.add_float("reg_alpha", 0.0, 1.0)
        xgb_space.add_float("reg_lambda", 0.0, 1.0)
        spaces["xgboost"] = xgb_space
    
    # LightGBM
    if HAS_LGB:
        lgb_space = ParameterSpace()
        lgb_space.add_int("n_estimators", 50, 500)
        lgb_space.add_int("max_depth", 3, 15)
        lgb_space.add_float("learning_rate", 0.01, 0.3, log=True)
        lgb_space.add_float("subsample", 0.6, 1.0)
        lgb_space.add_float("colsample_bytree", 0.6, 1.0)
        lgb_space.add_int("num_leaves", 10, 300)
        lgb_space.add_float("reg_alpha", 0.0, 1.0)
        lgb_space.add_float("reg_lambda", 0.0, 1.0)
        spaces["lightgbm"] = lgb_space
    
    return spaces


def optimize_fusion_model(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[OptimizationConfig] = None
) -> OptimizationResult:
    """
    Optimize fusion prediction model.
    
    Args:
        model_class: Model class to optimize.
        X: Training features.
        y: Training targets.
        config: Optimization configuration.
        
    Returns:
        Optimization result.
    """
    if config is None:
        config = OptimizationConfig()
    
    # Get appropriate parameter space
    spaces = create_model_parameter_spaces()
    
    # Determine model type
    model_name = model_class.__name__.lower()
    if "randomforest" in model_name:
        param_space = spaces.get("random_forest")
    elif "xgb" in model_name:
        param_space = spaces.get("xgboost")
    elif "lgb" in model_name or "lightgbm" in model_name:
        param_space = spaces.get("lightgbm")
    else:
        # Create default parameter space
        param_space = ParameterSpace()
        param_space.add_int("n_estimators", 50, 200)
        param_space.add_int("max_depth", 3, 15)
    
    if param_space is None:
        raise ValueError(f"No parameter space defined for {model_name}")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    result = optimizer.optimize_model(model_class, param_space, X, y)
    
    return result