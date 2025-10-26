# Contributing to Nuclear Fusion Analyzer ML Model

Thank you for your interest in contributing to the Nuclear Fusion Analyzer ML Model! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, gender identity, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)
- Basic understanding of machine learning and nuclear fusion physics (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/nuclear-fusion-analyzer.git
cd nuclear-fusion-analyzer
```

3. Add the original repository as upstream:
```bash
git remote add upstream https://github.com/originaluser/nuclear-fusion-analyzer.git
```

## Development Setup

### Local Development Environment

1. **Create virtual environment**:
```bash
python -m venv fusion_env
source fusion_env/bin/activate  # On Windows: fusion_env\Scripts\activate
```

2. **Install dependencies**:
```bash
make install-dev
# or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

3. **Set up pre-commit hooks**:
```bash
make setup
# or manually:
pre-commit install
```

4. **Verify installation**:
```bash
make test
python main.py --help
```

### Docker Development Environment

```bash
# Build development container
docker-compose --profile dev build

# Start development environment
docker-compose --profile dev up

# Run tests in container
docker-compose exec api pytest
```

### Environment Configuration

Create a `.env` file for local development:
```bash
FUSION_CONFIG_ENV=development
FUSION_LOG_LEVEL=DEBUG
FUSION_API_HOST=localhost
FUSION_API_PORT=8000
```

## Coding Standards

### Code Style

We follow PEP 8 guidelines with some project-specific conventions:

#### Python Code Style
- **Line length**: Maximum 88 characters (Black default)
- **Imports**: Use absolute imports, group them using isort
- **Naming**: 
  - Variables and functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

#### Type Hints
All public functions and methods must include type hints:
```python
def predict_q_factor(
    self, 
    magnetic_field: float, 
    plasma_current: float,
    **kwargs: Any
) -> Dict[str, float]:
    """Predict Q factor from plasma parameters."""
    pass
```

#### Docstrings
Use Google-style docstrings for all public functions and classes:
```python
def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic fusion dataset.
    
    Args:
        n_samples: Number of samples to generate.
        
    Returns:
        DataFrame containing fusion plasma parameters and Q factor.
        
    Raises:
        ValueError: If n_samples is not positive.
        
    Example:
        >>> generator = FusionDataGenerator()
        >>> data = generator.generate_dataset(n_samples=500)
        >>> print(data.shape)
        (500, 8)
    """
    pass
```

### Code Quality Tools

Use the provided automation:
```bash
make format        # Format code with black and isort
make lint          # Run flake8, black, isort checks
make type-check    # Run mypy type checking
make security      # Run bandit security analysis
```

Manual usage:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
bandit -r src/
```

### Configuration

- Black: `.black` settings in `pyproject.toml`
- isort: Settings in `pyproject.toml`
- flake8: Settings in `.flake8`
- mypy: Settings in `mypy.ini`

## Testing Guidelines

### Test Structure

```
tests/
├── test_data/              # Data generation and processing tests
├── test_models/            # Model training and prediction tests
├── test_api/              # API endpoint tests
├── test_utils/            # Utility function tests
├── test_integration/      # End-to-end integration tests
├── conftest.py           # Shared fixtures
└── test_config.py        # Test configuration
```

### Writing Tests

#### Unit Tests
Test individual functions and methods in isolation:
```python
import pytest
from src.data.generator import FusionDataGenerator

class TestFusionDataGenerator:
    """Test cases for FusionDataGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return FusionDataGenerator(random_state=42)
    
    def test_generate_dataset_default(self, generator):
        """Test dataset generation with default parameters."""
        data = generator.generate_dataset()
        
        assert len(data) == 1000  # default n_samples
        assert 'q_factor' in data.columns
        assert data['q_factor'].min() >= 0
        assert not data.isnull().any().any()
    
    def test_generate_dataset_custom_size(self, generator):
        """Test dataset generation with custom size."""
        data = generator.generate_dataset(n_samples=500)
        assert len(data) == 500
    
    @pytest.mark.parametrize("n_samples", [0, -1, -100])
    def test_generate_dataset_invalid_size(self, generator, n_samples):
        """Test dataset generation with invalid sizes."""
        with pytest.raises(ValueError):
            generator.generate_dataset(n_samples=n_samples)
```

#### Integration Tests
Test complete workflows:
```python
def test_complete_training_pipeline():
    """Test complete model training pipeline."""
    # Generate data
    generator = FusionDataGenerator(random_state=42)
    data = generator.generate_dataset(n_samples=1000)
    
    # Process data
    processor = FusionDataProcessor()
    processed = processor.preprocess_pipeline(data, target_column='q_factor')
    
    # Train model
    predictor = FusionPredictor()
    results = predictor.train_model('random_forest', 
                                   processed['X_train'], processed['y_train'],
                                   processed['X_val'], processed['y_val'])
    
    # Verify results
    assert results['val_r2'] > 0.8  # Expect good performance
    assert 'model' in results
    assert predictor.models['random_forest'] is not None
```

#### API Tests
Test all API endpoints:
```python
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

class TestPredictionAPI:
    """Test prediction API endpoints."""
    
    def test_predict_endpoint(self):
        """Test single prediction endpoint."""
        payload = {
            "magnetic_field": 5.3,
            "plasma_current": 15.0,
            "electron_density": 1.0e20,
            "ion_temperature": 20.0,
            "electron_temperature": 15.0,
            "neutral_beam_power": 50.0,
            "rf_heating_power": 30.0
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "q_factor" in data
        assert isinstance(data["q_factor"], float)
        assert data["q_factor"] > 0
```

### Test Configuration

#### Fixtures
Define reusable test components in `conftest.py`:
```python
import pytest
import pandas as pd
from src.data.generator import FusionDataGenerator

@pytest.fixture(scope="session")
def sample_data():
    """Generate sample data for testing."""
    generator = FusionDataGenerator(random_state=42)
    return generator.generate_dataset(n_samples=100)

@pytest.fixture
def mock_model(mocker):
    """Mock trained model for testing."""
    mock = mocker.Mock()
    mock.predict.return_value = [1.5, 2.0, 0.8]
    return mock
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test categories
pytest tests/test_models/ -v
pytest tests/test_api/ -v

# Run with markers
pytest -m unit
pytest -m integration
pytest -m slow

# Parallel execution
pytest -n auto

# Debug mode
pytest --pdb
```

### Test Markers

Use pytest markers to categorize tests:
```python
@pytest.mark.unit
def test_data_validation():
    """Unit test for data validation."""
    pass

@pytest.mark.integration
def test_api_model_integration():
    """Integration test for API and model."""
    pass

@pytest.mark.slow
def test_large_dataset_training():
    """Slow test for large dataset training."""
    pass
```

### Coverage Requirements

- **Minimum coverage**: 90% for all new code
- **Critical components**: 95% coverage required
- **Integration tests**: Must cover all major workflows

## Submitting Changes

### Branch Naming

Use descriptive branch names following this pattern:
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

Examples:
- `feature/add-lstm-model`
- `bugfix/fix-api-validation`
- `docs/update-readme`

### Commit Messages

Follow conventional commit format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build/tooling changes

Examples:
```
feat(models): add LSTM model for time series prediction

Implement LSTM neural network for temporal fusion data analysis.
Includes proper sequence preprocessing and validation.

Closes #123
```

```
fix(api): resolve validation error for negative parameters

Add proper bounds checking for plasma parameters in API endpoints.
Negative values now return appropriate error messages.

Fixes #456
```

### Pull Request Process

1. **Create feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes following coding standards**

3. **Add tests for new functionality**

4. **Run quality checks**:
```bash
make lint
make test
make type-check
```

5. **Update documentation if needed**

6. **Commit changes with clear messages**

7. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

8. **Create pull request with description**:
   - Describe the changes made
   - Link to related issues
   - Include testing information
   - Add screenshots for UI changes

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] No merge conflicts

## Related Issues
Closes #123
Fixes #456
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and quality checks
2. **Code review**: Maintainers review code for quality and compliance
3. **Discussion**: Address feedback and make requested changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge or rebase and merge based on change type

## Issue Reporting

### Bug Reports

Use the bug report template:
```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g. macOS 12.0]
- Python: [e.g. 3.9.7]
- Version: [e.g. 1.0.0]

**Additional Context**
Any other relevant information.
```

### Performance Issues

Include:
- Performance metrics (timing, memory usage)
- Dataset size and characteristics
- Hardware specifications
- Profiling information if available

## Feature Requests

Use the feature request template:
```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches considered.

**Additional Context**
Any other relevant information.
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: OpenAPI/Swagger specifications
3. **User Documentation**: README, tutorials, examples
4. **Developer Documentation**: Contributing, architecture
5. **Release Documentation**: Changelog, migration guides

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up to date with code changes
- Use proper markdown formatting
- Include diagrams where helpful

### Building Documentation

```bash
make docs         # Generate documentation
make docs-serve   # Serve documentation locally
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Email**: [project-email@example.com] for private matters

### Getting Help

- Check existing documentation
- Search existing issues
- Ask in GitHub Discussions
- Contact maintainers directly for security issues

### Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Annual contributor summary

Thank you for contributing to the Nuclear Fusion Analyzer ML Model project!