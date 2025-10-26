# Model Deployment Utilities Documentation

This directory contains comprehensive model deployment utilities for the Nuclear Fusion Analyzer ML platform.

## Overview

The deployment utilities provide end-to-end capabilities for packaging, deploying, versioning, and managing ML models in production environments.

## Components

### 1. Core Deployment Infrastructure (`src/utils/deployment_utils.py`)

**ModelPackager**
- Packages trained models with all dependencies
- Creates deployment-ready containers with Dockerfile and API wrapper
- Generates deployment scripts and configuration files
- Calculates checksums and metadata for integrity verification

**ContainerDeployer**
- Builds Docker images from model packages
- Deploys to Kubernetes clusters with scaling configuration
- Manages container lifecycle and health checks
- Supports rolling updates and rollback strategies

**CloudDeployer**
- Deploys models to AWS SageMaker endpoints
- Supports Azure ML and Google Cloud AI Platform
- Handles cloud-specific configuration and scaling
- Manages endpoint lifecycle and monitoring

### 2. Deployment Automation (`scripts/deploy_model.py`)

**Command-Line Interface**
```bash
# Deploy to Docker
python scripts/deploy_model.py --model-path ./models/fusion_model.joblib --platform docker

# Deploy to Kubernetes
python scripts/deploy_model.py --config deployment_config.yaml --platform kubernetes

# Deploy to AWS SageMaker
python scripts/deploy_model.py --model-id fusion-v1.0 --platform aws-sagemaker
```

**Features**
- Multiple deployment platforms (Docker, Kubernetes, AWS SageMaker)
- Configurable deployment parameters
- Automatic model packaging and containerization
- Deployment manifest generation
- Health check integration

### 3. Model Registry (`scripts/model_registry.py`)

**Version Management**
```bash
# Register new model
python scripts/model_registry.py register --model-path ./models/fusion_model.joblib --version 1.0.0 --name "Fusion Predictor V1"

# List models
python scripts/model_registry.py list --status active

# Compare versions
python scripts/model_registry.py compare --model-id fusion-analyzer --version1 1.0.0 --version2 1.1.0

# Rollback to previous version
python scripts/model_registry.py rollback --model-id fusion-analyzer --version 1.0.0
```

**Capabilities**
- SQLite-based model metadata storage
- Version lineage tracking with parent-child relationships
- Performance metrics comparison between versions
- Model status management (active, deprecated, archived)
- A/B testing experiment framework
- Automated rollback functionality

### 4. Production Server Management (`scripts/server_manager.py`)

**Server Operations**
```bash
# Start server with multiple workers
python scripts/server_manager.py start --port 8000 --workers 4

# Check server status
python scripts/server_manager.py status

# Scale workers dynamically
python scripts/server_manager.py scale --workers 8

# Graceful shutdown
python scripts/server_manager.py stop --graceful
```

**Advanced Features**
- Multi-worker process management with automatic scaling
- Real-time health monitoring and automatic recovery
- Resource usage tracking (CPU, memory)
- Graceful shutdown with configurable timeouts
- Process restart on failure detection
- Signal handling for production environments

## Deployment Workflows

### Basic Model Deployment

1. **Package Model**
   ```python
   from src.utils.deployment_utils import create_deployment_system
   
   packager, container_deployer, cloud_deployer = create_deployment_system()
   
   package = packager.package_model(
       predictor=trained_model,
       model_id="fusion-analyzer",
       version="1.0.0",
       output_dir="./deployments"
   )
   ```

2. **Build Container**
   ```python
   image_name = container_deployer.build_docker_image(
       package_path="./deployments/fusion-analyzer_1.0.0",
       image_name="fusion-analyzer/model",
       image_tag="1.0.0"
   )
   ```

3. **Deploy to Platform**
   ```python
   # For Kubernetes
   deployment_result = container_deployer.deploy_to_kubernetes(
       image_name=image_name,
       deployment_config=deployment_config
   )
   ```

### Production Deployment Pipeline

1. **Model Registration**
   - Register model in version control system
   - Generate metadata and performance metrics
   - Create deployment artifacts

2. **Automated Testing**
   - Run integration tests on packaged model
   - Perform performance benchmarks
   - Validate API endpoints

3. **Staged Deployment**
   - Deploy to staging environment
   - Run acceptance tests
   - Monitor performance metrics

4. **Production Rollout**
   - Blue-green deployment strategy
   - Gradual traffic shifting
   - Real-time monitoring and alerting

## Configuration

### Deployment Configuration Example

```yaml
environment: production
target_platform: kubernetes
scaling_config:
  replicas: 3
  app_name: fusion-analyzer
  namespace: production
  service_type: LoadBalancer
resource_limits:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
health_check:
  path: /health
  initial_delay: 30
  period: 10
monitoring:
  enabled: true
  prometheus: true
rollback_config:
  enabled: true
  strategy: RollingUpdate
```

### Server Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 30,
    "keepalive": 5
  },
  "monitoring": {
    "health_check_interval": 30,
    "restart_on_failure": true,
    "max_memory_mb": 1024,
    "max_cpu_percent": 80
  }
}
```

## Monitoring and Observability

### Health Checks
- Process-level health monitoring
- Resource usage tracking
- API endpoint availability
- Model performance metrics

### Logging and Metrics
- Structured logging with configurable levels
- Performance metrics collection
- Error tracking and alerting
- Deployment audit trails

### Dashboard Integration
- Real-time server status monitoring
- Model performance visualization
- Resource utilization graphs
- Deployment history tracking

## Security Considerations

### Container Security
- Non-root user execution
- Minimal base images
- Security scanning integration
- Resource limits enforcement

### API Security
- Authentication and authorization
- Rate limiting and throttling
- Input validation and sanitization
- HTTPS/TLS encryption

### Model Security
- Model artifact integrity verification
- Access control and permissions
- Audit logging for model access
- Secure model storage

## Best Practices

### Development
- Use consistent versioning schemes (semantic versioning)
- Maintain comprehensive model metadata
- Implement thorough testing at all levels
- Document deployment procedures

### Operations
- Monitor model performance continuously
- Implement automated rollback procedures
- Use blue-green deployments for zero downtime
- Maintain deployment environment parity

### Maintenance
- Regular security updates
- Performance optimization monitoring
- Capacity planning and scaling
- Disaster recovery planning

## Troubleshooting

### Common Issues
- Port conflicts during multi-worker deployment
- Resource exhaustion under high load
- Model loading failures
- Network connectivity issues

### Diagnostic Tools
- Health check endpoints
- Process monitoring utilities
- Log aggregation and analysis
- Performance profiling tools

## Integration Examples

### CI/CD Pipeline Integration
```yaml
# GitHub Actions example
deploy:
  runs-on: ubuntu-latest
  steps:
    - name: Deploy Model
      run: |
        python scripts/deploy_model.py \
          --model-path ${{ env.MODEL_PATH }} \
          --platform kubernetes \
          --config deploy_config.yaml
```

### Monitoring Integration
```python
# Prometheus metrics integration
from prometheus_client import Counter, Histogram, Gauge

prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
```

This deployment system provides enterprise-grade capabilities for managing ML models in production environments with comprehensive automation, monitoring, and reliability features.