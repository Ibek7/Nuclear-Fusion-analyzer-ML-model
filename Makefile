# Makefile for Nuclear Fusion Analyzer
.PHONY: help install test lint format clean run-api run-jupyter docker-build docker-run

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy

# Default target
help:
	@echo "Nuclear Fusion Analyzer - Available Commands:"
	@echo "============================================="
	@echo "Development:"
	@echo "  install          Install dependencies"
	@echo "  test             Run unit tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  lint             Run linting (flake8, mypy)"
	@echo "  format           Format code with black"
	@echo "  clean            Clean temporary files"
	@echo ""
	@echo "Application:"
	@echo "  run-api          Start FastAPI server"
	@echo "  run-jupyter      Start Jupyter Lab"
	@echo "  run-streamlit    Start Streamlit dashboard"
	@echo "  run-main         Run main fusion analyzer"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"
	@echo "  docker-dev       Run development container"
	@echo "  docker-jupyter   Run Jupyter in Docker"
	@echo ""
	@echo "Data & Models:"
	@echo "  generate-data    Generate sample dataset"
	@echo "  train-models     Train all models"
	@echo "  evaluate-models  Evaluate model performance"
	@echo ""
	@echo "Utilities:"
	@echo "  setup-git        Setup git hooks"
	@echo "  setup-env        Setup development environment"

# Development commands
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

lint:
	$(FLAKE8) src/ api/ tests/
	$(MYPY) src/ api/ --ignore-missing-imports

format:
	$(BLACK) src/ api/ tests/ --line-length 100

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Application commands
run-api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

run-jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

run-main:
	$(PYTHON) main.py

# Docker commands
docker-build:
	docker build -t fusion-analyzer .

docker-run:
	docker run -p 8000:8000 fusion-analyzer

docker-dev:
	docker-compose --profile development up

docker-jupyter:
	docker-compose --profile jupyter up jupyter

# Data and model commands
generate-data:
	$(PYTHON) -c "from src.data.generator import FusionDataGenerator; \
	              gen = FusionDataGenerator(); \
	              data = gen.generate_dataset(10000); \
	              data.to_parquet('data/sample_data.parquet'); \
	              print('Generated sample dataset')"

train-models:
	$(PYTHON) -c "from main import FusionAnalyzer; \
	              analyzer = FusionAnalyzer(); \
	              analyzer.train_models(); \
	              print('Trained all models')"

evaluate-models:
	$(PYTHON) -c "from main import FusionAnalyzer; \
	              analyzer = FusionAnalyzer(); \
	              analyzer.evaluate_models(); \
	              print('Evaluated all models')"

# Setup commands
setup-git:
	pre-commit install
	@echo "Git hooks installed"

setup-env:
	@echo "Setting up development environment..."
	$(MAKE) install
	$(MAKE) setup-git
	mkdir -p data logs saved_models results plots
	@echo "Development environment ready!"

# CI/CD commands
ci-test:
	$(PYTEST) tests/ --cov=src --cov-report=xml --cov-fail-under=80

ci-lint:
	$(FLAKE8) src/ api/ tests/ --max-line-length=100 --exit-zero
	$(MYPY) src/ api/ --ignore-missing-imports --strict

# Performance testing
benchmark:
	$(PYTHON) -m pytest tests/test_performance/ -v --benchmark-only

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/

# Database commands (if using database)
db-setup:
	$(PYTHON) -c "from scripts.setup_database import setup; setup()"

db-migrate:
	alembic upgrade head

# Deployment commands
deploy-staging:
	@echo "Deploying to staging..."
	# Add deployment commands here

deploy-prod:
	@echo "Deploying to production..."
	# Add production deployment commands here

# Quick development workflow
dev: clean format lint test
	@echo "Development workflow complete!"

# Full CI workflow
ci: clean ci-lint ci-test
	@echo "CI workflow complete!"

# Release workflow
release: clean format lint test docker-build
	@echo "Release workflow complete!"