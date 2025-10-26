# Nuclear Fusion Analyzer ML Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning platform for nuclear fusion plasma analysis, prediction, and optimization. This project provides advanced tools for analyzing fusion reactor performance, predicting Q factors, detecting anomalies, and optimizing operational parameters.

## 🚀 Features

### Core ML Capabilities
- **Q Factor Prediction**: Multiple ML models (Random Forest, Gradient Boosting, SVR, Neural Networks, LSTM)
- **Anomaly Detection**: Real-time detection of plasma disruptions and unusual behavior
- **Time Series Analysis**: Temporal modeling of fusion plasma evolution
- **Parameter Optimization**: Multi-objective optimization for reactor performance
- **Uncertainty Quantification**: Monte Carlo analysis for risk assessment

### Advanced Analysis
- **Sensitivity Analysis**: Identify critical control parameters
- **Pareto Front Analysis**: Trade-off optimization between performance and power consumption
- **Scenario Modeling**: Compare different operational strategies
- **Physics-based Validation**: Incorporate fusion physics constraints
- **Real-time Monitoring**: Continuous plasma state assessment

### Engineering Features
- **REST API**: FastAPI-based service for model deployment
- **Containerization**: Docker support for scalable deployment
- **Configuration Management**: Flexible YAML-based configuration system
- **Comprehensive Testing**: Full pytest test suite with >90% coverage
- **Interactive Notebooks**: Jupyter notebooks for exploration and analysis
- **Automated Workflows**: Makefile automation for development tasks

## 🛠 Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- (Optional) Docker for containerized deployment

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/nuclear-fusion-analyzer.git
cd nuclear-fusion-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the analyzer
python main.py --model random_forest --n_samples 5000 --plot
```

### Using Make (Recommended)
```bash
make install          # Install all dependencies
make setup           # Complete setup including pre-commit hooks
make run-api         # Start the API server
make jupyter         # Launch Jupyter notebooks
```

## 🚀 Quick Start

### 1. Basic Usage
```python
from src.data.generator import FusionDataGenerator
from src.models.fusion_predictor import FusionPredictor
from src.data.processor import FusionDataProcessor

# Generate synthetic fusion data
generator = FusionDataGenerator(random_state=42)
data = generator.generate_dataset(n_samples=1000)

# Process and train
processor = FusionDataProcessor()
processed = processor.preprocess_pipeline(data, target_column='q_factor')

predictor = FusionPredictor()
results = predictor.train_model('random_forest', 
                               processed['X_train'], processed['y_train'],
                               processed['X_val'], processed['y_val'])

print(f"Model R²: {results['val_r2']:.4f}")
```

### 2. API Usage
```bash
# Start API server
make run-api

# Make predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"magnetic_field": 5.3, "plasma_current": 15.0, ...}'
```

### 3. Interactive Analysis
```bash
# Launch Jupyter notebooks
make jupyter

# Open notebooks/01_exploratory_analysis.ipynb
```

## 📁 Project Structure

```
nuclear-fusion-analyzer/
├── src/                          # Source code
│   ├── data/                    # Data generation and processing
│   ├── models/                  # Machine learning models
│   ├── utils/                   # Utilities and configuration
│   └── visualization/           # Plotting and visualization
├── api/                         # REST API
├── config/                      # Configuration files
├── tests/                       # Test suite
├── notebooks/                   # Jupyter notebooks
├── Dockerfile                   # Docker container definition
├── docker-compose.yml          # Multi-service deployment
├── Makefile                     # Development automation
└── main.py                      # Main application entry point
```

## 📚 API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints
- `POST /predict` - Predict Q factor from plasma parameters
- `POST /predict/batch` - Batch prediction for multiple parameter sets
- `POST /anomaly/detect` - Detect anomalies in plasma data
- `POST /optimize` - Optimize parameters for target Q factor

## 🐳 Docker Deployment

```bash
# Single container
docker build -t fusion-analyzer .
docker run -p 8000:8000 fusion-analyzer

# Multi-service with docker-compose
docker-compose --profile prod up    # Production environment
docker-compose --profile dev up     # Development environment
docker-compose --profile jupyter up # Jupyter notebook server
```

## 🧪 Testing

```bash
# Run all tests
make test

# With coverage
make test-coverage

# Specific test categories
pytest tests/test_models/
pytest tests/test_api/
```

## 📊 Model Performance

| Model | R² Score | MAE | RMSE | Training Time |
|-------|----------|-----|------|---------------|
| Random Forest | 0.924 | 0.158 | 0.203 | 2.3s |
| Gradient Boosting | 0.919 | 0.162 | 0.209 | 4.1s |
| Neural Network | 0.902 | 0.175 | 0.230 | 12.4s |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Add tests for new functionality
4. Run the test suite: `make test`
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- **ITER Organization** - For fusion physics insights and data structures
- **scikit-learn community** - For excellent machine learning tools
- **FastAPI team** - For the modern API framework

---

**Note**: This is a research and educational project. For production use in fusion facilities, additional validation and safety considerations are required.