"""
Model interpretability and explainability tools for nuclear fusion ML models.

This module provides comprehensive model interpretation capabilities
including SHAP values, LIME explanations, feature importance analysis,
and model-agnostic explanation techniques for fusion prediction models.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    # SHAP for model explanations
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    # LIME for local interpretability
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    # Permutation importance
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN_INSPECTION = True
except ImportError:
    HAS_SKLEARN_INSPECTION = False

try:
    # Additional interpretation tools
    from sklearn.tree import export_text
    from sklearn.ensemble import RandomForestRegressor
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Container for feature importance results."""
    
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_std: Optional[np.ndarray]
    method: str
    metadata: Dict[str, Any]


@dataclass
class ShapExplanation:
    """Container for SHAP explanation results."""
    
    shap_values: np.ndarray
    base_value: Union[float, np.ndarray]
    feature_names: List[str]
    data: np.ndarray
    model_output: np.ndarray
    explanation_type: str  # 'global', 'local', 'summary'


@dataclass
class LimeExplanation:
    """Container for LIME explanation results."""
    
    instance_id: int
    feature_contributions: Dict[str, float]
    prediction: float
    intercept: float
    local_accuracy: float
    explanation_fit: Any


class ModelInterpreter:
    """
    Comprehensive model interpretation system for fusion ML models.
    
    Provides multiple interpretation methods including SHAP, LIME,
    permutation importance, and custom physics-aware explanations.
    """
    
    def __init__(self, model, feature_names: List[str], config: Optional[Dict] = None):
        """
        Initialize the model interpreter.
        
        Args:
            model: Trained ML model to interpret.
            feature_names: List of feature names.
            config: Configuration dictionary.
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        
        # Interpretation parameters
        self.max_display_features = self.config.get('max_display_features', 20)
        self.shap_sample_size = self.config.get('shap_sample_size', 100)
        self.lime_sample_size = self.config.get('lime_sample_size', 5000)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Physics-aware feature groups for fusion domain
        self.physics_groups = {
            'magnetic_confinement': ['magnetic_field', 'q_factor', 'plasma_current'],
            'thermal_properties': ['ion_temperature', 'electron_temperature', 'pressure'],
            'density_profile': ['electron_density', 'density_peaking', 'line_density'],
            'heating_systems': ['neutral_beam_power', 'rf_heating_power', 'ohmic_power'],
            'stability_metrics': ['beta', 'confinement_time', 'energy_content'],
            'performance_indicators': ['fusion_power', 'q_factor', 'gain_factor']
        }
        
        logger.info(f"ModelInterpreter initialized for model: {type(model).__name__}")
    
    def calculate_feature_importance(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     method: str = 'permutation') -> FeatureImportance:
        """
        Calculate feature importance using various methods.
        
        Args:
            X: Feature matrix.
            y: Target values.
            method: Importance calculation method.
            
        Returns:
            FeatureImportance object.
        """
        if method == 'permutation' and HAS_SKLEARN_INSPECTION:
            return self._permutation_importance(X, y)
        elif method == 'tree_based' and hasattr(self.model, 'feature_importances_'):
            return self._tree_based_importance()
        elif method == 'coefficients' and hasattr(self.model, 'coef_'):
            return self._coefficient_importance()
        elif method == 'shap_global' and HAS_SHAP:
            return self._shap_global_importance(X)
        else:
            raise ValueError(f"Unsupported importance method: {method}")
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> FeatureImportance:
        """Calculate permutation-based feature importance."""
        try:
            result = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_squared_error'
            )
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=result.importances_mean,
                importance_std=result.importances_std,
                method='permutation',
                metadata={'n_repeats': 10}
            )
            
        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {e}")
            raise
    
    def _tree_based_importance(self) -> FeatureImportance:
        """Calculate tree-based feature importance."""
        importance_scores = self.model.feature_importances_
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            importance_std=None,
            method='tree_based',
            metadata={'model_type': type(self.model).__name__}
        )
    
    def _coefficient_importance(self) -> FeatureImportance:
        """Calculate coefficient-based importance for linear models."""
        coefficients = np.abs(self.model.coef_)
        
        # Normalize coefficients
        normalized_coef = coefficients / np.sum(coefficients)
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=normalized_coef,
            importance_std=None,
            method='coefficients',
            metadata={'model_type': type(self.model).__name__}
        )
    
    def _shap_global_importance(self, X: np.ndarray) -> FeatureImportance:
        """Calculate SHAP-based global feature importance."""
        if not HAS_SHAP:
            raise ImportError("SHAP is required for SHAP-based importance")
        
        try:
            # Initialize SHAP explainer if not done
            if self.shap_explainer is None:
                self._initialize_shap_explainer(X)
            
            # Calculate SHAP values for sample
            sample_size = min(self.shap_sample_size, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values as importance
            if isinstance(shap_values, list):
                # Multi-output case
                importance_scores = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
            else:
                importance_scores = np.mean(np.abs(shap_values), axis=0)
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=importance_scores,
                importance_std=None,
                method='shap_global',
                metadata={'sample_size': sample_size}
            )
            
        except Exception as e:
            logger.error(f"SHAP global importance calculation failed: {e}")
            raise
    
    def explain_with_shap(self, 
                          X: np.ndarray, 
                          explanation_type: str = 'summary',
                          instance_idx: Optional[int] = None) -> ShapExplanation:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X: Feature matrix.
            explanation_type: Type of explanation ('summary', 'local', 'global').
            instance_idx: Instance index for local explanations.
            
        Returns:
            ShapExplanation object.
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required for SHAP explanations")
        
        try:
            # Initialize SHAP explainer if not done
            if self.shap_explainer is None:
                self._initialize_shap_explainer(X)
            
            # Calculate SHAP values
            if explanation_type == 'local' and instance_idx is not None:
                X_explain = X[instance_idx:instance_idx+1]
            else:
                sample_size = min(self.shap_sample_size, len(X))
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_explain = X[sample_indices]
            
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Get model predictions
            model_output = self.model.predict(X_explain)
            
            # Get base value
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
            else:
                base_value = np.mean(model_output)
            
            return ShapExplanation(
                shap_values=shap_values,
                base_value=base_value,
                feature_names=self.feature_names,
                data=X_explain,
                model_output=model_output,
                explanation_type=explanation_type
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise
    
    def explain_with_lime(self, 
                          X: np.ndarray, 
                          instance_idx: int,
                          mode: str = 'regression') -> LimeExplanation:
        """
        Generate LIME explanation for a specific instance.
        
        Args:
            X: Feature matrix.
            instance_idx: Index of instance to explain.
            mode: LIME mode ('regression' or 'classification').
            
        Returns:
            LimeExplanation object.
        """
        if not HAS_LIME:
            raise ImportError("LIME is required for LIME explanations")
        
        try:
            # Initialize LIME explainer if not done
            if self.lime_explainer is None:
                self._initialize_lime_explainer(X, mode)
            
            # Get instance to explain
            instance = X[instance_idx]
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            # Extract feature contributions
            feature_contributions = dict(explanation.as_list())
            
            # Get prediction for this instance
            prediction = self.model.predict(instance.reshape(1, -1))[0]
            
            return LimeExplanation(
                instance_id=instance_idx,
                feature_contributions=feature_contributions,
                prediction=prediction,
                intercept=explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0,
                local_accuracy=explanation.score if hasattr(explanation, 'score') else 0.0,
                explanation_fit=explanation
            )
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            raise
    
    def _initialize_shap_explainer(self, X: np.ndarray):
        """Initialize SHAP explainer based on model type."""
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(self.model, 'estimators_') or 'forest' in str(type(self.model)).lower():
                self.shap_explainer = shap.TreeExplainer(self.model)
            # Try LinearExplainer for linear models
            elif hasattr(self.model, 'coef_'):
                self.shap_explainer = shap.LinearExplainer(self.model, X)
            # Use KernelExplainer as fallback
            else:
                background_size = min(100, len(X))
                background = shap.kmeans(X, background_size)
                self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
            
            logger.info(f"SHAP explainer initialized: {type(self.shap_explainer).__name__}")
            
        except Exception as e:
            logger.error(f"SHAP explainer initialization failed: {e}")
            raise
    
    def _initialize_lime_explainer(self, X: np.ndarray, mode: str = 'regression'):
        """Initialize LIME explainer."""
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                mode=mode,
                training_labels=None,
                random_state=42
            )
            
            logger.info("LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"LIME explainer initialization failed: {e}")
            raise
    
    def analyze_physics_groups(self, importance: FeatureImportance) -> Dict[str, float]:
        """
        Analyze feature importance by physics-based groups.
        
        Args:
            importance: FeatureImportance object.
            
        Returns:
            Dictionary of group importance scores.
        """
        group_importance = {}
        
        for group_name, group_features in self.physics_groups.items():
            group_score = 0.0
            group_count = 0
            
            for feature in group_features:
                if feature in importance.feature_names:
                    idx = importance.feature_names.index(feature)
                    group_score += importance.importance_scores[idx]
                    group_count += 1
            
            if group_count > 0:
                group_importance[group_name] = group_score / group_count
            else:
                group_importance[group_name] = 0.0
        
        return group_importance
    
    def generate_interpretation_report(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray,
                                       sample_instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation report.
        
        Args:
            X: Feature matrix.
            y: Target values.
            sample_instance_idx: Instance index for local explanations.
            
        Returns:
            Comprehensive interpretation report.
        """
        report = {}
        
        try:
            # Feature importance analysis
            importance_methods = ['permutation']
            if hasattr(self.model, 'feature_importances_'):
                importance_methods.append('tree_based')
            if hasattr(self.model, 'coef_'):
                importance_methods.append('coefficients')
            if HAS_SHAP:
                importance_methods.append('shap_global')
            
            importance_results = {}
            for method in importance_methods:
                try:
                    importance = self.calculate_feature_importance(X, y, method)
                    importance_results[method] = importance
                    
                    # Physics group analysis
                    group_analysis = self.analyze_physics_groups(importance)
                    importance_results[f"{method}_groups"] = group_analysis
                    
                except Exception as e:
                    logger.warning(f"Importance method {method} failed: {e}")
            
            report['feature_importance'] = importance_results
            
            # SHAP analysis
            if HAS_SHAP:
                try:
                    shap_summary = self.explain_with_shap(X, 'summary')
                    report['shap_summary'] = shap_summary
                    
                    if sample_instance_idx is not None:
                        shap_local = self.explain_with_shap(X, 'local', sample_instance_idx)
                        report['shap_local'] = shap_local
                        
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")
            
            # LIME analysis
            if HAS_LIME and sample_instance_idx is not None:
                try:
                    lime_explanation = self.explain_with_lime(X, sample_instance_idx)
                    report['lime_local'] = lime_explanation
                    
                except Exception as e:
                    logger.warning(f"LIME analysis failed: {e}")
            
            # Model-specific analysis
            if hasattr(self.model, 'estimators_'):
                # Ensemble model analysis
                report['ensemble_analysis'] = self._analyze_ensemble_model()
            
            # Physics-based insights
            report['physics_insights'] = self._generate_physics_insights(importance_results)
            
            logger.info("Interpretation report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def _analyze_ensemble_model(self) -> Dict[str, Any]:
        """Analyze ensemble model properties."""
        analysis = {}
        
        if hasattr(self.model, 'estimators_'):
            n_estimators = len(self.model.estimators_)
            analysis['n_estimators'] = n_estimators
            
            # Tree depth analysis for forest models
            if hasattr(self.model.estimators_[0], 'tree_'):
                depths = [est.tree_.max_depth for est in self.model.estimators_]
                analysis['tree_depths'] = {
                    'mean': np.mean(depths),
                    'std': np.std(depths),
                    'min': np.min(depths),
                    'max': np.max(depths)
                }
        
        return analysis
    
    def _generate_physics_insights(self, importance_results: Dict) -> Dict[str, Any]:
        """Generate physics-based insights from importance analysis."""
        insights = {}
        
        # Find most important physics group
        if 'permutation_groups' in importance_results:
            groups = importance_results['permutation_groups']
            most_important_group = max(groups, key=groups.get)
            insights['most_important_physics_group'] = most_important_group
            insights['group_rankings'] = sorted(groups.items(), key=lambda x: x[1], reverse=True)
        
        # Check for physics consistency
        consistency_checks = {}
        
        # Check if magnetic confinement features are important
        if 'permutation' in importance_results:
            importance = importance_results['permutation']
            magnetic_features = ['magnetic_field', 'q_factor', 'plasma_current']
            magnetic_importance = [
                importance.importance_scores[importance.feature_names.index(f)]
                for f in magnetic_features if f in importance.feature_names
            ]
            
            if magnetic_importance:
                consistency_checks['magnetic_confinement_importance'] = np.mean(magnetic_importance)
        
        insights['physics_consistency'] = consistency_checks
        
        return insights


def create_model_interpreter(model, 
                             feature_names: List[str],
                             config_path: Optional[str] = None) -> ModelInterpreter:
    """
    Create a model interpreter with configuration.
    
    Args:
        model: Trained ML model.
        feature_names: List of feature names.
        config_path: Path to configuration file.
        
    Returns:
        Configured ModelInterpreter.
    """
    config = {}
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config().get('interpretability', {})
    
    return ModelInterpreter(model, feature_names, config)