"""
Data validation and quality assessment for nuclear fusion analysis.

This module provides comprehensive data validation, quality checks, and 
preprocessing validation for fusion plasma parameters and ML model inputs.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
import re

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Container for validation issues."""
    
    severity: ValidationSeverity
    field: str
    message: str
    value: Optional[Any] = None
    expected_range: Optional[Tuple[float, float]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Container for validation results."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    cleaned_data: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = field(default_factory=dict)


class FusionDataValidator:
    """
    Comprehensive data validator for nuclear fusion parameters.
    
    Validates plasma parameters against physics constraints, detects outliers,
    checks data quality, and provides suggestions for data cleaning.
    """
    
    # Physics-based parameter constraints
    PARAMETER_CONSTRAINTS = {
        'magnetic_field': {
            'min': 0.1,    # Minimum practical field strength (T)
            'max': 20.0,   # Maximum achievable field strength (T)
            'typical_min': 2.0,
            'typical_max': 8.0,
            'units': 'Tesla',
            'description': 'Toroidal magnetic field strength'
        },
        'plasma_current': {
            'min': 0.1,    # Minimum plasma current (MA)
            'max': 50.0,   # Maximum practical current (MA)
            'typical_min': 5.0,
            'typical_max': 25.0,
            'units': 'MA',
            'description': 'Plasma current'
        },
        'electron_density': {
            'min': 1e18,   # Minimum density (m^-3)
            'max': 5e21,   # Maximum density (m^-3)
            'typical_min': 5e19,
            'typical_max': 2e20,
            'units': 'm^-3',
            'description': 'Electron density'
        },
        'ion_temperature': {
            'min': 0.1,    # Minimum temperature (keV)
            'max': 200.0,  # Maximum temperature (keV)
            'typical_min': 5.0,
            'typical_max': 50.0,
            'units': 'keV',
            'description': 'Ion temperature'
        },
        'electron_temperature': {
            'min': 0.1,    # Minimum temperature (keV)
            'max': 200.0,  # Maximum temperature (keV)
            'typical_min': 3.0,
            'typical_max': 40.0,
            'units': 'keV',
            'description': 'Electron temperature'
        },
        'neutral_beam_power': {
            'min': 0.0,    # Can be zero (no NBI)
            'max': 500.0,  # Maximum NBI power (MW)
            'typical_min': 10.0,
            'typical_max': 150.0,
            'units': 'MW',
            'description': 'Neutral beam injection power'
        },
        'rf_heating_power': {
            'min': 0.0,    # Can be zero (no RF)
            'max': 200.0,  # Maximum RF power (MW)
            'typical_min': 5.0,
            'typical_max': 80.0,
            'units': 'MW',
            'description': 'Radio frequency heating power'
        },
        'q_factor': {
            'min': 0.001,  # Very low gain
            'max': 100.0,  # Theoretical maximum
            'typical_min': 0.1,
            'typical_max': 20.0,
            'units': 'dimensionless',
            'description': 'Fusion gain factor'
        },
        'beta': {
            'min': 0.0,    # Minimum beta
            'max': 0.2,    # Maximum stable beta (20%)
            'typical_min': 0.01,
            'typical_max': 0.1,
            'units': 'dimensionless',
            'description': 'Plasma beta (pressure/magnetic pressure)'
        },
        'confinement_time': {
            'min': 0.001,  # Minimum confinement (s)
            'max': 10.0,   # Maximum practical confinement (s)
            'typical_min': 0.1,
            'typical_max': 3.0,
            'units': 's',
            'description': 'Energy confinement time'
        }
    }
    
    # Physics relationships for cross-validation
    PHYSICS_RELATIONSHIPS = {
        'temperature_ratio': {
            'description': 'Ion/electron temperature ratio should be reasonable',
            'min_ratio': 0.5,   # Ti/Te >= 0.5
            'max_ratio': 3.0,   # Ti/Te <= 3.0
        },
        'density_greenwald': {
            'description': 'Density should not exceed Greenwald limit',
            'safety_factor': 1.2,  # 20% margin
        },
        'beta_limit': {
            'description': 'Beta should not exceed Troyon limit',
            'safety_factor': 1.1,  # 10% margin
        }
    }
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize the fusion data validator.
        
        Args:
            validation_level: Strictness of validation checks.
        """
        self.validation_level = validation_level
        self.custom_constraints: Dict[str, Dict] = {}
        
        logger.info(f"FusionDataValidator initialized with {validation_level.value} validation")
    
    def validate_dataframe(
        self, 
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        allow_missing: bool = True
    ) -> ValidationResult:
        """
        Validate an entire DataFrame of fusion parameters.
        
        Args:
            data: DataFrame to validate.
            required_columns: List of required column names.
            allow_missing: Whether to allow missing values.
            
        Returns:
            ValidationResult: Comprehensive validation results.
        """
        issues = []
        is_valid = True
        
        # Check for empty data
        if data.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="data",
                message="DataFrame is empty",
                suggestion="Provide non-empty dataset"
            ))
            return ValidationResult(is_valid=False, issues=issues)
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field="columns",
                    message=f"Missing required columns: {missing_cols}",
                    suggestion="Add missing columns to dataset"
                ))
                is_valid = False
        
        # Validate each column
        cleaned_data = data.copy()
        
        for column in data.columns:
            if column in self.PARAMETER_CONSTRAINTS or column in self.custom_constraints:
                column_issues, cleaned_column = self._validate_column(
                    data[column], column, allow_missing
                )
                issues.extend(column_issues)
                
                if cleaned_column is not None:
                    cleaned_data[column] = cleaned_column
        
        # Physics relationship validation
        physics_issues = self._validate_physics_relationships(data)
        issues.extend(physics_issues)
        
        # Statistical outlier detection
        outlier_issues = self._detect_statistical_outliers(data)
        issues.extend(outlier_issues)
        
        # Data quality checks
        quality_issues = self._check_data_quality(data)
        issues.extend(quality_issues)
        
        # Determine overall validity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            is_valid = False
        
        # Generate summary
        summary = self._generate_validation_summary(data, issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_data=cleaned_data,
            summary=summary
        )
    
    def validate_single_record(
        self, 
        record: Dict[str, float],
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a single fusion parameter record.
        
        Args:
            record: Dictionary of parameter values.
            required_fields: List of required field names.
            
        Returns:
            ValidationResult: Validation results for the record.
        """
        # Convert to DataFrame for consistent validation
        df = pd.DataFrame([record])
        return self.validate_dataframe(df, required_fields, allow_missing=False)
    
    def _validate_column(
        self, 
        series: pd.Series, 
        column_name: str,
        allow_missing: bool
    ) -> Tuple[List[ValidationIssue], Optional[pd.Series]]:
        """Validate a single column against constraints."""
        issues = []
        cleaned_series = series.copy()
        
        # Get constraints
        constraints = self.PARAMETER_CONSTRAINTS.get(
            column_name, 
            self.custom_constraints.get(column_name, {})
        )
        
        if not constraints:
            return issues, None
        
        # Check for missing values
        missing_count = series.isnull().sum()
        if missing_count > 0:
            if not allow_missing:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=column_name,
                    message=f"{missing_count} missing values found",
                    suggestion="Remove or impute missing values"
                ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=column_name,
                    message=f"{missing_count} missing values found",
                    suggestion="Consider imputation if needed"
                ))
        
        # Validate non-missing values
        valid_values = series.dropna()
        if len(valid_values) == 0:
            return issues, cleaned_series
        
        # Range validation
        if 'min' in constraints:
            below_min = valid_values < constraints['min']
            if below_min.any():
                count = below_min.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=column_name,
                    message=f"{count} values below minimum ({constraints['min']})",
                    expected_range=(constraints['min'], constraints.get('max', float('inf'))),
                    suggestion=f"Ensure {column_name} >= {constraints['min']} {constraints.get('units', '')}"
                ))
                
                # Clip values if lenient validation
                if self.validation_level == ValidationLevel.LENIENT:
                    cleaned_series = cleaned_series.clip(lower=constraints['min'])
        
        if 'max' in constraints:
            above_max = valid_values > constraints['max']
            if above_max.any():
                count = above_max.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=column_name,
                    message=f"{count} values above maximum ({constraints['max']})",
                    expected_range=(constraints.get('min', 0), constraints['max']),
                    suggestion=f"Ensure {column_name} <= {constraints['max']} {constraints.get('units', '')}"
                ))
                
                # Clip values if lenient validation
                if self.validation_level == ValidationLevel.LENIENT:
                    cleaned_series = cleaned_series.clip(upper=constraints['max'])
        
        # Typical range warnings
        if 'typical_min' in constraints and 'typical_max' in constraints:
            outside_typical = (
                (valid_values < constraints['typical_min']) | 
                (valid_values > constraints['typical_max'])
            )
            if outside_typical.any():
                count = outside_typical.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=column_name,
                    message=f"{count} values outside typical range "
                           f"({constraints['typical_min']}-{constraints['typical_max']})",
                    suggestion=f"Verify {column_name} values are realistic for fusion conditions"
                ))
        
        # Check for negative values where not expected
        if constraints.get('min', 0) >= 0:
            negative_values = valid_values < 0
            if negative_values.any():
                count = negative_values.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=column_name,
                    message=f"{count} negative values found",
                    suggestion=f"{column_name} should be non-negative"
                ))
        
        return issues, cleaned_series
    
    def _validate_physics_relationships(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate physics-based relationships between parameters."""
        issues = []
        
        # Temperature ratio check
        if 'ion_temperature' in data.columns and 'electron_temperature' in data.columns:
            temp_ratio = data['ion_temperature'] / data['electron_temperature']
            rel = self.PHYSICS_RELATIONSHIPS['temperature_ratio']
            
            invalid_ratio = (
                (temp_ratio < rel['min_ratio']) | 
                (temp_ratio > rel['max_ratio'])
            )
            
            if invalid_ratio.any():
                count = invalid_ratio.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="temperature_ratio",
                    message=f"{count} records with unusual Ti/Te ratio",
                    expected_range=(rel['min_ratio'], rel['max_ratio']),
                    suggestion="Check temperature measurements for consistency"
                ))
        
        # Greenwald density limit check
        if all(col in data.columns for col in ['electron_density', 'plasma_current', 'magnetic_field']):
            # Simplified Greenwald limit: n_e <= I_p / (π * a²) where a ≈ R/3
            # Using approximate major radius R ≈ 6m, minor radius a ≈ 2m
            a_minor = 2.0  # meters
            greenwald_limit = data['plasma_current'] * 1e6 / (np.pi * a_minor**2) * 1e20  # m^-3
            
            rel = self.PHYSICS_RELATIONSHIPS['density_greenwald']
            above_greenwald = data['electron_density'] > greenwald_limit * rel['safety_factor']
            
            if above_greenwald.any():
                count = above_greenwald.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="electron_density",
                    message=f"{count} records above Greenwald density limit",
                    suggestion="High density may lead to disruptions"
                ))
        
        # Beta limit check (simplified Troyon limit)
        if 'beta' in data.columns and 'plasma_current' in data.columns and 'magnetic_field' in data.columns:
            # Troyon limit: β ≤ βN * I / (a * B) where βN ≈ 2.8 for typical plasmas
            beta_n = 2.8e-2  # Normalized beta limit
            a_minor = 2.0    # meters
            
            troyon_limit = beta_n * data['plasma_current'] / (a_minor * data['magnetic_field'])
            
            rel = self.PHYSICS_RELATIONSHIPS['beta_limit']
            above_troyon = data['beta'] > troyon_limit * rel['safety_factor']
            
            if above_troyon.any():
                count = above_troyon.sum()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="beta",
                    message=f"{count} records above Troyon beta limit",
                    suggestion="High beta may lead to MHD instabilities"
                ))
        
        return issues
    
    def _detect_statistical_outliers(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detect statistical outliers using multiple methods."""
        issues = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in self.PARAMETER_CONSTRAINTS:
                continue
                
            series = data[column].dropna()
            if len(series) < 10:  # Need sufficient data for outlier detection
                continue
            
            outlier_indices = set()
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                iqr_outliers = (
                    (series < Q1 - 1.5 * IQR) | 
                    (series > Q3 + 1.5 * IQR)
                )
                outlier_indices.update(series[iqr_outliers].index)
            
            # Z-score method (for normally distributed data)
            if len(series) > 30:  # Only for larger samples
                z_scores = np.abs(stats.zscore(series))
                z_outliers = z_scores > 3.0
                outlier_indices.update(series[z_outliers].index)
            
            # Modified Z-score using median (more robust)
            median = series.median()
            mad = np.median(np.abs(series - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (series - median) / mad
                modified_z_outliers = np.abs(modified_z_scores) > 3.5
                outlier_indices.update(series[modified_z_outliers].index)
            
            # Report outliers
            if outlier_indices:
                count = len(outlier_indices)
                severity = ValidationSeverity.WARNING
                
                # Escalate severity if many outliers
                if count > len(series) * 0.1:  # More than 10% outliers
                    severity = ValidationSeverity.CRITICAL
                
                issues.append(ValidationIssue(
                    severity=severity,
                    field=column,
                    message=f"{count} statistical outliers detected",
                    suggestion="Review outlier values for data entry errors"
                ))
        
        return issues
    
    def _check_data_quality(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Perform general data quality checks."""
        issues = []
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="duplicates",
                message=f"{duplicate_count} duplicate rows found",
                suggestion="Remove or investigate duplicate records"
            ))
        
        # Check for constant columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if data[column].nunique() <= 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=column,
                    message="Column has constant values",
                    suggestion="Verify if constant values are expected"
                ))
        
        # Check data types
        for column in data.columns:
            if column in self.PARAMETER_CONSTRAINTS:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        field=column,
                        message="Non-numeric data type",
                        suggestion="Convert to numeric data type"
                    ))
        
        # Check for infinite values
        for column in numeric_columns:
            inf_count = np.isinf(data[column]).sum()
            if inf_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=column,
                    message=f"{inf_count} infinite values found",
                    suggestion="Replace infinite values with appropriate bounds"
                ))
        
        return issues
    
    def _generate_validation_summary(
        self, 
        data: pd.DataFrame, 
        issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        summary = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'total_issues': len(issues),
            'issue_breakdown': {
                'critical': len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]),
                'warning': len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
                'info': len([i for i in issues if i.severity == ValidationSeverity.INFO])
            },
            'columns_with_issues': len(set(issue.field for issue in issues)),
            'missing_data_summary': {
                col: data[col].isnull().sum() 
                for col in data.columns 
                if data[col].isnull().sum() > 0
            },
            'data_ranges': {
                col: {
                    'min': float(data[col].min()) if pd.api.types.is_numeric_dtype(data[col]) else None,
                    'max': float(data[col].max()) if pd.api.types.is_numeric_dtype(data[col]) else None,
                    'mean': float(data[col].mean()) if pd.api.types.is_numeric_dtype(data[col]) else None
                }
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col])
            }
        }
        
        return summary
    
    def add_custom_constraint(
        self, 
        column_name: str, 
        constraints: Dict[str, Any]
    ):
        """
        Add custom validation constraints for a parameter.
        
        Args:
            column_name: Name of the parameter column.
            constraints: Dictionary of constraints (min, max, typical_min, typical_max, etc.).
        """
        self.custom_constraints[column_name] = constraints
        logger.info(f"Added custom constraints for {column_name}")
    
    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_result: Results from validation.
            
        Returns:
            Formatted validation report string.
        """
        report = []
        report.append("=" * 50)
        report.append("FUSION DATA VALIDATION REPORT")
        report.append("=" * 50)
        
        # Summary
        summary = validation_result.summary
        report.append(f"\nDataset Summary:")
        report.append(f"  Total Records: {summary.get('total_records', 'N/A')}")
        report.append(f"  Total Columns: {summary.get('total_columns', 'N/A')}")
        report.append(f"  Validation Status: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        
        # Issue breakdown
        breakdown = summary.get('issue_breakdown', {})
        report.append(f"\nIssue Summary:")
        report.append(f"  Critical Issues: {breakdown.get('critical', 0)}")
        report.append(f"  Warnings: {breakdown.get('warning', 0)}")
        report.append(f"  Information: {breakdown.get('info', 0)}")
        report.append(f"  Total Issues: {summary.get('total_issues', 0)}")
        
        # Detailed issues
        if validation_result.issues:
            report.append(f"\nDetailed Issues:")
            
            for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                severity_issues = [i for i in validation_result.issues if i.severity == severity]
                
                if severity_issues:
                    report.append(f"\n{severity.value.upper()} Issues:")
                    
                    for issue in severity_issues:
                        report.append(f"  • Field: {issue.field}")
                        report.append(f"    Message: {issue.message}")
                        
                        if issue.expected_range:
                            report.append(f"    Expected Range: {issue.expected_range}")
                        
                        if issue.suggestion:
                            report.append(f"    Suggestion: {issue.suggestion}")
                        
                        report.append("")
        
        # Data ranges
        data_ranges = summary.get('data_ranges', {})
        if data_ranges:
            report.append(f"\nData Ranges:")
            for col, ranges in data_ranges.items():
                if ranges['min'] is not None:
                    report.append(f"  {col}: {ranges['min']:.3f} - {ranges['max']:.3f} "
                                f"(mean: {ranges['mean']:.3f})")
        
        # Missing data
        missing_data = summary.get('missing_data_summary', {})
        if missing_data:
            report.append(f"\nMissing Data:")
            for col, count in missing_data.items():
                report.append(f"  {col}: {count} missing values")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


def validate_fusion_parameters(
    data: Union[pd.DataFrame, Dict[str, float]],
    validation_level: ValidationLevel = ValidationLevel.MODERATE,
    required_fields: Optional[List[str]] = None
) -> ValidationResult:
    """
    Convenience function for validating fusion parameters.
    
    Args:
        data: DataFrame or dictionary of fusion parameters.
        validation_level: Strictness of validation.
        required_fields: List of required field names.
        
    Returns:
        ValidationResult: Comprehensive validation results.
    """
    validator = FusionDataValidator(validation_level)
    
    if isinstance(data, dict):
        return validator.validate_single_record(data, required_fields)
    else:
        return validator.validate_dataframe(data, required_fields)