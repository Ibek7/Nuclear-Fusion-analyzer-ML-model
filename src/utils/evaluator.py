"""
Model evaluation and validation framework for fusion analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix, roc_auc_score,
        precision_recall_curve, roc_curve
    )
    from sklearn.model_selection import cross_val_score, validation_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Evaluation metrics limited.")
    SKLEARN_AVAILABLE = False


class FusionModelEvaluator:
    """
    Comprehensive evaluation framework for fusion prediction models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.evaluation_results = {}
        self.benchmarks = self._set_fusion_benchmarks()
    
    def _set_fusion_benchmarks(self) -> Dict[str, float]:
        """
        Set benchmark values for fusion parameters.
        
        Returns:
            Dictionary of benchmark values
        """
        return {
            'q_factor_breakeven': 1.0,
            'q_factor_ignition': 5.0,
            'lawson_criterion_ignition': 1e21,  # m^-3 * s
            'triple_product_ignition': 3e21,    # keV * m^-3 * s
            'beta_limit': 0.1,
            'minimum_confinement_time': 0.1,    # seconds
            'maximum_disruption_rate': 0.05     # 5% disruption rate
        }
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = 'Unknown') -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of regression metrics
        """
        if SKLEARN_AVAILABLE:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        else:
            # Manual calculation if sklearn not available
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'r2': float(r2)
            }
        
        # Additional custom metrics
        metrics.update({
            'mape': float(self._calculate_mape(y_true, y_pred)),
            'max_error': float(np.max(np.abs(y_true - y_pred))),
            'median_error': float(np.median(np.abs(y_true - y_pred))),
            'model_name': model_name
        })
        
        # Physics-based evaluation
        if model_name.lower() in ['q_factor', 'fusion_efficiency']:
            metrics.update(self._evaluate_fusion_physics(y_true, y_pred))
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _evaluate_fusion_physics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions from fusion physics perspective.
        
        Args:
            y_true: True Q factors
            y_pred: Predicted Q factors
            
        Returns:
            Physics-based evaluation metrics
        """
        physics_metrics = {}
        
        # Breakeven prediction accuracy
        true_breakeven = y_true >= self.benchmarks['q_factor_breakeven']
        pred_breakeven = y_pred >= self.benchmarks['q_factor_breakeven']
        
        if true_breakeven.sum() > 0:
            physics_metrics['breakeven_accuracy'] = float(
                np.mean(true_breakeven == pred_breakeven)
            )
        
        # Ignition prediction accuracy
        true_ignition = y_true >= self.benchmarks['q_factor_ignition']
        pred_ignition = y_pred >= self.benchmarks['q_factor_ignition']
        
        if true_ignition.sum() > 0:
            physics_metrics['ignition_accuracy'] = float(
                np.mean(true_ignition == pred_ignition)
            )
        
        # Physical constraint violations
        unphysical_predictions = (y_pred < 0) | (y_pred > 100)  # Q > 100 is unrealistic
        physics_metrics['physical_constraint_violations'] = float(
            unphysical_predictions.sum() / len(y_pred)
        )
        
        return physics_metrics
    
    def evaluate_anomaly_detection(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            y_true: True anomaly labels (0: normal, 1: anomaly)
            y_pred: Predicted anomaly labels
            y_scores: Anomaly scores (optional)
            
        Returns:
            Anomaly detection metrics
        """
        if not SKLEARN_AVAILABLE:
            # Basic accuracy calculation
            accuracy = np.mean(y_true == y_pred)
            return {'accuracy': float(accuracy)}
        
        # Convert to binary format if needed
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        metrics = {
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'f1_score': 0.0,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = float(
                2 * metrics['precision'] * metrics['recall'] / 
                (metrics['precision'] + metrics['recall'])
            )
        
        # AUC if scores available
        if y_scores is not None:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true_binary, y_scores))
            except ValueError:
                metrics['auc_roc'] = 0.0
        
        # Fusion-specific metrics
        if 'disruption' in str(metrics).lower():
            metrics.update(self._evaluate_disruption_detection(y_true_binary, y_pred_binary))
        
        return metrics
    
    def _evaluate_disruption_detection(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate disruption detection from fusion safety perspective.
        
        Args:
            y_true: True disruption labels
            y_pred: Predicted disruption labels
            
        Returns:
            Disruption-specific metrics
        """
        metrics = {}
        
        # False negative rate (missed disruptions) - critical for safety
        fn_rate = np.mean((y_true == 1) & (y_pred == 0))
        metrics['missed_disruption_rate'] = float(fn_rate)
        
        # False positive rate (false alarms) - impacts operations
        fp_rate = np.mean((y_true == 0) & (y_pred == 1))
        metrics['false_alarm_rate'] = float(fp_rate)
        
        # Early warning capability (if time series data)
        if len(y_true) > 10:  # Minimum length for time series analysis
            # Calculate detection lead time (simplified)
            disruption_indices = np.where(y_true == 1)[0]
            if len(disruption_indices) > 0:
                # Check if predictions occur before actual disruptions
                early_detections = 0
                for idx in disruption_indices:
                    if idx > 0 and y_pred[idx-1] == 1:
                        early_detections += 1
                
                metrics['early_detection_rate'] = float(
                    early_detections / len(disruption_indices)
                )
        
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: ML model to evaluate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available for cross-validation'}
        
        try:
            # Regression metrics
            cv_scores = {}
            
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                cv_scores[f'{metric}_mean'] = float(np.mean(scores))
                cv_scores[f'{metric}_std'] = float(np.std(scores))
            
            # Convert negative scores to positive
            if 'neg_mean_squared_error_mean' in cv_scores:
                cv_scores['mse_mean'] = -cv_scores['neg_mean_squared_error_mean']
                cv_scores['mse_std'] = cv_scores['neg_mean_squared_error_std']
            
            if 'neg_mean_absolute_error_mean' in cv_scores:
                cv_scores['mae_mean'] = -cv_scores['neg_mean_absolute_error_mean']
                cv_scores['mae_std'] = cv_scores['neg_mean_absolute_error_std']
            
            return cv_scores
            
        except Exception as e:
            return {'error': str(e)}
    
    def learning_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                              train_sizes: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze learning curves to diagnose bias/variance.
        
        Args:
            model: ML model to analyze
            X: Feature matrix
            y: Target vector
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Learning curve analysis results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available for learning curves'}
        
        try:
            from sklearn.model_selection import learning_curve
            
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            results = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': (-train_scores.mean(axis=1)).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': (-val_scores.mean(axis=1)).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
            
            # Diagnose overfitting/underfitting
            final_train_score = results['train_scores_mean'][-1]
            final_val_score = results['val_scores_mean'][-1]
            score_gap = final_train_score - final_val_score
            
            if score_gap > 0.1:  # Significant gap
                results['diagnosis'] = 'overfitting'
            elif final_val_score > 0.8:  # Good validation performance
                results['diagnosis'] = 'good_fit'
            else:
                results['diagnosis'] = 'underfitting'
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def feature_importance_analysis(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract and analyze feature importance.
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
            
        Returns:
            Feature importance dictionary
        """
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_)
                if coefficients.ndim > 1:
                    coefficients = coefficients.flatten()
                importance_dict = dict(zip(feature_names, coefficients))
                
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {
                    k: v/total_importance for k, v in importance_dict.items()
                }
            
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
        
        return importance_dict
    
    def model_stability_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                               n_bootstraps: int = 100) -> Dict[str, Any]:
        """
        Analyze model stability using bootstrap sampling.
        
        Args:
            model: ML model to analyze
            X: Feature matrix
            y: Target vector
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            Stability analysis results
        """
        predictions = []
        scores = []
        
        try:
            for i in range(n_bootstraps):
                # Bootstrap sampling
                n_samples = len(X)
                indices = np.random.choice(n_samples, n_samples, replace=True)
                
                X_bootstrap = X.iloc[indices]
                y_bootstrap = y.iloc[indices]
                
                # Train model on bootstrap sample
                model_copy = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                model_copy.fit(X_bootstrap, y_bootstrap)
                
                # Make predictions on original data
                y_pred = model_copy.predict(X)
                predictions.append(y_pred)
                
                # Calculate score
                if SKLEARN_AVAILABLE:
                    score = r2_score(y, y_pred)
                else:
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                scores.append(score)
            
            # Analyze prediction stability
            predictions = np.array(predictions)
            prediction_std = np.std(predictions, axis=0)
            
            results = {
                'mean_score': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'score_stability': float(1.0 - np.std(scores) / (np.mean(scores) + 1e-8)),
                'prediction_stability': float(1.0 - np.mean(prediction_std)),
                'confidence_intervals': {
                    'score_95_ci': [
                        float(np.percentile(scores, 2.5)),
                        float(np.percentile(scores, 97.5))
                    ]
                }
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                               y_pred: np.ndarray = None,
                               model_name: str = 'Unknown') -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test targets
            y_pred: Predictions (optional, will compute if not provided)
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation results
        """
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        results = {
            'model_name': model_name,
            'test_samples': len(y_test)
        }
        
        # Basic regression metrics
        results['regression_metrics'] = self.evaluate_regression_model(
            y_test.values, y_pred, model_name
        )
        
        # Cross-validation
        if len(X_test) > 10:  # Minimum samples for CV
            results['cross_validation'] = self.cross_validate_model(
                model, X_test, y_test
            )
        
        # Feature importance
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            results['feature_importance'] = self.feature_importance_analysis(
                model, X_test.columns.tolist()
            )
        
        # Learning curve analysis
        if len(X_test) > 50:  # Minimum samples for learning curves
            results['learning_curve'] = self.learning_curve_analysis(
                model, X_test, y_test
            )
        
        # Model stability
        if len(X_test) > 20:  # Minimum samples for stability analysis
            results['stability_analysis'] = self.model_stability_analysis(
                model, X_test, y_test, n_bootstraps=20
            )
        
        # Physics-based validation
        results['physics_validation'] = self._validate_physics_constraints(
            y_test.values, y_pred
        )
        
        return results
    
    def _validate_physics_constraints(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Validate predictions against physics constraints.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Physics validation results
        """
        validation = {
            'constraint_violations': 0,
            'physical_realism_score': 1.0
        }
        
        # Check for negative Q factors (unphysical)
        negative_q = np.sum(y_pred < 0)
        if negative_q > 0:
            validation['negative_q_predictions'] = int(negative_q)
            validation['constraint_violations'] += negative_q
        
        # Check for extremely high Q factors (unrealistic)
        extreme_q = np.sum(y_pred > 50)  # Q > 50 is very rare
        if extreme_q > 0:
            validation['extreme_q_predictions'] = int(extreme_q)
            validation['constraint_violations'] += extreme_q
        
        # Calculate physical realism score
        total_violations = validation['constraint_violations']
        validation['physical_realism_score'] = float(
            max(0, 1.0 - total_violations / len(y_pred))
        )
        
        return validation
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append(f"MODEL EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Model: {evaluation_results.get('model_name', 'Unknown')}")
        report.append(f"Test Samples: {evaluation_results.get('test_samples', 0)}")
        report.append("")
        
        # Regression metrics
        if 'regression_metrics' in evaluation_results:
            metrics = evaluation_results['regression_metrics']
            report.append("REGRESSION METRICS:")
            report.append(f"  R² Score: {metrics.get('r2', 0):.4f}")
            report.append(f"  RMSE: {metrics.get('rmse', 0):.4f}")
            report.append(f"  MAE: {metrics.get('mae', 0):.4f}")
            report.append(f"  MAPE: {metrics.get('mape', 0):.2f}%")
            report.append("")
        
        # Physics validation
        if 'physics_validation' in evaluation_results:
            physics = evaluation_results['physics_validation']
            report.append("PHYSICS VALIDATION:")
            report.append(f"  Physical Realism Score: {physics.get('physical_realism_score', 0):.4f}")
            report.append(f"  Constraint Violations: {physics.get('constraint_violations', 0)}")
            report.append("")
        
        # Cross-validation
        if 'cross_validation' in evaluation_results:
            cv = evaluation_results['cross_validation']
            if 'r2_mean' in cv:
                report.append("CROSS-VALIDATION:")
                report.append(f"  CV R² Mean: {cv.get('r2_mean', 0):.4f} ± {cv.get('r2_std', 0):.4f}")
                report.append("")
        
        # Feature importance (top 5)
        if 'feature_importance' in evaluation_results:
            importance = evaluation_results['feature_importance']
            report.append("TOP 5 IMPORTANT FEATURES:")
            for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                report.append(f"  {i+1}. {feature}: {imp:.4f}")
            report.append("")
        
        # Model diagnosis
        if 'learning_curve' in evaluation_results:
            diagnosis = evaluation_results['learning_curve'].get('diagnosis', 'unknown')
            report.append("MODEL DIAGNOSIS:")
            report.append(f"  Learning Status: {diagnosis}")
            report.append("")
        
        return "\n".join(report)