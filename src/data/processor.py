"""
Data preprocessing and feature engineering for fusion data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class FusionDataProcessor:
    """
    Data processor for nuclear fusion datasets with preprocessing,
    feature engineering, and data validation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.feature_selector = None
        self.pca = None
        self.imputer = None
        
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate fusion data for physical consistency and quality.
        
        Args:
            data: Input fusion dataset
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_samples': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'physical_violations': [],
            'data_quality_score': 0.0
        }
        
        # Check for physical violations
        if 'plasma_temperature' in data.columns:
            # Temperature should be positive and within fusion range
            temp_violations = (data['plasma_temperature'] <= 0) | \
                            (data['plasma_temperature'] > 1e9)
            if temp_violations.any():
                report['physical_violations'].append(
                    f"Temperature violations: {temp_violations.sum()} samples"
                )
        
        if 'plasma_density' in data.columns:
            # Density should be positive
            density_violations = data['plasma_density'] <= 0
            if density_violations.any():
                report['physical_violations'].append(
                    f"Density violations: {density_violations.sum()} samples"
                )
        
        if 'q_factor' in data.columns:
            # Q factor should be positive
            q_violations = data['q_factor'] <= 0
            if q_violations.any():
                report['physical_violations'].append(
                    f"Q factor violations: {q_violations.sum()} samples"
                )
        
        # Calculate data quality score
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        duplicate_ratio = report['duplicate_rows'] / len(data)
        violation_ratio = len(report['physical_violations']) / len(data.columns)
        
        report['data_quality_score'] = 1.0 - (missing_ratio + duplicate_ratio + violation_ratio)
        
        return report
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean fusion data by handling missing values and outliers.
        
        Args:
            data: Input dataset
            
        Returns:
            Cleaned dataset
        """
        cleaned_data = data.copy()
        
        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Handle missing values
        numerical_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        
        # Impute numerical columns
        if len(numerical_columns) > 0:
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                cleaned_data[numerical_columns] = self.imputer.fit_transform(
                    cleaned_data[numerical_columns]
                )
            else:
                cleaned_data[numerical_columns] = self.imputer.transform(
                    cleaned_data[numerical_columns]
                )
        
        # Impute categorical columns
        for col in categorical_columns:
            mode_value = cleaned_data[col].mode()
            if len(mode_value) > 0:
                cleaned_data[col].fillna(mode_value[0], inplace=True)
        
        # Remove outliers using IQR method
        for col in numerical_columns:
            if col in cleaned_data.columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)
        
        return cleaned_data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features from existing fusion parameters.
        
        Args:
            data: Input dataset
            
        Returns:
            Dataset with engineered features
        """
        engineered_data = data.copy()
        
        # Derived physics parameters
        if all(col in data.columns for col in ['plasma_temperature', 'plasma_density']):
            # Thermal energy density
            engineered_data['thermal_energy_density'] = (
                engineered_data['plasma_temperature'] * 
                engineered_data['plasma_density'] * 1.38e-23
            )
        
        if all(col in data.columns for col in ['magnetic_field', 'plasma_density']):
            # Magnetic pressure
            engineered_data['magnetic_pressure'] = (
                engineered_data['magnetic_field']**2 / (2 * 4e-7 * np.pi)
            )
            
            # Plasma frequency
            engineered_data['plasma_frequency'] = np.sqrt(
                engineered_data['plasma_density'] * 1.6e-19**2 / 
                (8.85e-12 * 9.11e-31)
            )
        
        if all(col in data.columns for col in ['plasma_temperature', 'magnetic_field']):
            # Gyrofrequency
            engineered_data['gyrofrequency'] = (
                1.6e-19 * engineered_data['magnetic_field'] / (2 * 3.34e-27)
            )
            
            # Thermal velocity
            engineered_data['thermal_velocity'] = np.sqrt(
                2 * 1.38e-23 * engineered_data['plasma_temperature'] / 3.34e-27
            )
        
        # Performance ratios
        if all(col in data.columns for col in ['fusion_power', 'total_heating_power']):
            engineered_data['power_efficiency'] = (
                engineered_data['fusion_power'] / 
                (engineered_data['total_heating_power'] + 1e-6)
            )
        
        if all(col in data.columns for col in ['lawson_criterion']):
            # Normalized Lawson criterion
            engineered_data['normalized_lawson'] = (
                engineered_data['lawson_criterion'] / 1e21
            )
        
        # Stability indicators
        if all(col in data.columns for col in ['beta_plasma', 'safety_factor']):
            # Troyon beta limit
            engineered_data['troyon_beta_ratio'] = (
                engineered_data['beta_plasma'] / 
                (0.028 * engineered_data['safety_factor'])
            )
        
        # Time-based features
        if 'time_in_shot' in data.columns and 'shot_duration' in data.columns:
            engineered_data['shot_progress'] = (
                engineered_data['time_in_shot'] / 
                engineered_data['shot_duration']
            )
        
        # System health composite score
        health_columns = [col for col in data.columns if 'health' in col]
        if health_columns:
            engineered_data['overall_system_health'] = (
                engineered_data[health_columns].mean(axis=1)
            )
        
        # Interaction features
        if all(col in data.columns for col in ['deuterium_fraction', 'tritium_fraction']):
            engineered_data['dt_ratio'] = (
                engineered_data['deuterium_fraction'] / 
                (engineered_data['tritium_fraction'] + 1e-6)
            )
        
        return engineered_data
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None,
                      scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Scaled training and test features
        """
        if scaler_type not in self.scalers:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == 'minmax':
                self.scalers[scaler_type] = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scalers[scaler_type] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler = self.scalers[scaler_type]
        
        # Fit and transform training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k: int = 20) -> pd.DataFrame:
        """
        Select top k features using univariate feature selection.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
            
        Returns:
            Selected features
        """
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def apply_pca(self, X: pd.DataFrame, 
                  n_components: float = 0.95) -> pd.DataFrame:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X: Input features
            n_components: Number of components or explained variance ratio
            
        Returns:
            PCA-transformed features
        """
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        
        # Create column names for PCA components
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    def create_polynomial_features(self, X: pd.DataFrame, 
                                 degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for capturing non-linear relationships.
        
        Args:
            X: Input features
            degree: Polynomial degree
            
        Returns:
            Dataset with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def preprocess_pipeline(self, data: pd.DataFrame, 
                          target_column: str = None,
                          test_size: float = 0.2,
                          validation_size: float = 0.1) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for fusion data.
        
        Args:
            data: Input dataset
            target_column: Name of target column
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            
        Returns:
            Dictionary with processed datasets and metadata
        """
        from sklearn.model_selection import train_test_split
        
        # Validate data
        validation_report = self.validate_data(data)
        
        # Clean data
        cleaned_data = self.clean_data(data)
        
        # Engineer features
        engineered_data = self.engineer_features(cleaned_data)
        
        # Separate features and target
        if target_column and target_column in engineered_data.columns:
            X = engineered_data.drop(columns=[target_column])
            y = engineered_data[target_column]
        else:
            X = engineered_data
            y = None
        
        # Split data
        if y is not None:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=42
            )
            X_temp = X_train
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val = train_test_split(
                X_temp, test_size=val_size_adjusted, random_state=42
            )
            y_train = y_val = y_test = None
        
        # Scale features
        X_train_scaled, X_val_scaled = self.scale_features(X_train, X_val)
        _, X_test_scaled = self.scale_features(X_train, X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': list(X_train_scaled.columns),
            'validation_report': validation_report,
            'data_shape': engineered_data.shape,
            'preprocessing_config': self.config
        }