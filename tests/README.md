# Unit Tests for Nuclear Fusion Analyzer

This directory contains comprehensive unit tests for the Nuclear Fusion Analyzer project.

## Test Structure

- `test_data/` - Tests for data generation and processing modules
- `test_models/` - Tests for machine learning models and prediction
- `test_utils/` - Tests for utility functions and evaluation metrics
- `test_visualization/` - Tests for plotting and visualization components

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_data/
pytest tests/test_models/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Requirements

- pytest
- pytest-cov
- numpy
- pandas
- scikit-learn

## Test Data

Tests use synthetic data generated specifically for testing purposes to ensure reproducible and reliable test results.