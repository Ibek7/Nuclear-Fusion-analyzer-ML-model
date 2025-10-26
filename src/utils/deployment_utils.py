"""
Advanced model deployment utilities for Nuclear Fusion Analyzer.

This module provides comprehensive deployment capabilities including
model packaging, containerization, cloud deployment, model versioning,
A/B testing, and production monitoring integration.
"""

import os
import json
import yaml
import logging
import tempfile
import zipfile
import shutil
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import subprocess

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    import kubernetes
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

import joblib
import numpy as np
import pandas as pd

from src.models.fusion_predictor import FusionPredictor
from src.utils.config_manager import ConfigManager
from src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ModelPackage:
    """Container for model package information."""
    
    model_id: str
    version: str
    model_type: str
    framework: str
    created_at: str
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    size_mb: float
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    
    environment: str
    target_platform: str
    scaling_config: Dict[str, Any]
    resource_limits: Dict[str, Any]
    health_check: Dict[str, Any]
    monitoring: Dict[str, Any]
    rollback_config: Dict[str, Any]


class ModelPackager:
    """
    Package ML models for deployment.
    
    Creates deployment-ready packages with models, dependencies,
    configuration, and metadata.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model packager.
        
        Args:
            config: Packaging configuration.
        """
        self.config = config or {}
        self.package_format = self.config.get('package_format', 'joblib')
        self.include_dependencies = self.config.get('include_dependencies', True)
        
        logger.info("ModelPackager initialized")
    
    def package_model(self, 
                      predictor: FusionPredictor,
                      model_id: str,
                      version: str,
                      output_dir: str,
                      metadata: Optional[Dict] = None) -> ModelPackage:
        """
        Package a trained model for deployment.
        
        Args:
            predictor: Trained fusion predictor.
            model_id: Unique model identifier.
            version: Model version.
            output_dir: Output directory for package.
            metadata: Additional metadata.
            
        Returns:
            ModelPackage information.
        """
        package_dir = Path(output_dir) / f"{model_id}_{version}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = package_dir / "model.joblib"
        joblib.dump(predictor, model_path)
        
        # Save model configuration
        model_config = {
            'model_id': model_id,
            'version': version,
            'model_type': 'fusion_predictor',
            'framework': 'scikit-learn',
            'created_at': datetime.now().isoformat(),
            'feature_names': getattr(predictor, 'feature_names_', []),
            'model_params': self._extract_model_params(predictor)
        }
        
        config_path = package_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Create requirements file
        if self.include_dependencies:
            requirements_path = package_dir / "requirements.txt"
            self._create_requirements_file(requirements_path)
        
        # Create deployment script
        deploy_script_path = package_dir / "deploy.py"
        self._create_deployment_script(deploy_script_path, model_config)
        
        # Create Docker file
        dockerfile_path = package_dir / "Dockerfile"
        self._create_dockerfile(dockerfile_path, model_config)
        
        # Create API wrapper
        api_path = package_dir / "api.py"
        self._create_api_wrapper(api_path, model_config)
        
        # Calculate package size and checksum
        package_size = self._calculate_directory_size(package_dir)
        package_checksum = self._calculate_directory_checksum(package_dir)
        
        # Create package archive
        archive_path = package_dir.parent / f"{model_id}_{version}.zip"
        self._create_archive(package_dir, archive_path)
        
        # Performance metrics (if available)
        performance_metrics = {}
        if hasattr(predictor, 'cv_results_'):
            performance_metrics = predictor.cv_results_
        
        # Create model package info
        model_package = ModelPackage(
            model_id=model_id,
            version=version,
            model_type='fusion_predictor',
            framework='scikit-learn',
            created_at=model_config['created_at'],
            performance_metrics=performance_metrics,
            dependencies=self._get_dependencies(),
            size_mb=package_size,
            checksum=package_checksum,
            metadata=metadata or {}
        )
        
        # Save package info
        package_info_path = package_dir / "package_info.json"
        with open(package_info_path, 'w') as f:
            json.dump(asdict(model_package), f, indent=2)
        
        logger.info(f"Model packaged successfully: {archive_path}")
        return model_package
    
    def _extract_model_params(self, predictor: FusionPredictor) -> Dict:
        """Extract model parameters."""
        params = {}
        
        for model_name, model in predictor.models.items():
            if hasattr(model, 'get_params'):
                params[model_name] = model.get_params()
        
        return params
    
    def _create_requirements_file(self, requirements_path: Path):
        """Create requirements.txt file."""
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "joblib>=1.0.0",
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0"
        ]
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_deployment_script(self, script_path: Path, model_config: Dict):
        """Create deployment script."""
        script_content = f'''#!/usr/bin/env python3
"""
Deployment script for {model_config['model_id']} v{model_config['version']}
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class ModelServer:
    def __init__(self, model_path="model.joblib"):
        self.model = joblib.load(model_path)
        self.model_config = {model_config}
    
    def predict(self, input_data):
        """Make predictions on input data."""
        if isinstance(input_data, dict):
            # Single prediction
            df = pd.DataFrame([input_data])
            return self.model.predict(df.values)[0]
        elif isinstance(input_data, list):
            # Batch prediction
            df = pd.DataFrame(input_data)
            return self.model.predict(df.values).tolist()
        else:
            # Direct array input
            return self.model.predict(input_data)
    
    def health_check(self):
        """Health check endpoint."""
        return {{
            "status": "healthy",
            "model_id": self.model_config["model_id"],
            "version": self.model_config["version"],
            "timestamp": pd.Timestamp.now().isoformat()
        }}

if __name__ == "__main__":
    server = ModelServer()
    print("Model server initialized successfully")
    print(f"Model: {{server.model_config['model_id']}} v{{server.model_config['version']}}")
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
    
    def _create_dockerfile(self, dockerfile_path: Path, model_config: Dict):
        """Create Dockerfile for containerized deployment."""
        dockerfile_content = f'''FROM python:3.9-slim

LABEL maintainer="Nuclear Fusion Analyzer Team"
LABEL model_id="{model_config['model_id']}"
LABEL version="{model_config['version']}"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "from api import app; import requests; requests.get('http://localhost:8000/health')"

# Start the API server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
    
    def _create_api_wrapper(self, api_path: Path, model_config: Dict):
        """Create FastAPI wrapper for the model."""
        api_content = f'''"""
FastAPI wrapper for {model_config['model_id']} v{model_config['version']}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="{model_config['model_id']} API",
    description="Nuclear Fusion Analyzer Model API",
    version="{model_config['version']}"
)

# Load model
model = joblib.load("model.joblib")
logger.info(f"Model loaded: {{model}}")

class PredictionInput(BaseModel):
    magnetic_field: float
    plasma_current: float
    electron_density: float
    ion_temperature: float
    electron_temperature: float
    neutral_beam_power: float
    rf_heating_power: float

class PredictionOutput(BaseModel):
    predictions: Dict[str, float]
    model_version: str = "{model_config['version']}"

class BatchPredictionInput(BaseModel):
    samples: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[Dict[str, float]]
    model_version: str = "{model_config['version']}"

@app.get("/health")
async def health_check():
    return {{
        "status": "healthy",
        "model_id": "{model_config['model_id']}",
        "version": "{model_config['version']}",
        "timestamp": pd.Timestamp.now().isoformat()
    }}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(df.values)[0]
        
        return PredictionOutput(
            predictions={{"q_factor": float(prediction)}}
        )
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    try:
        # Convert input to DataFrame
        samples = [sample.dict() for sample in input_data.samples]
        df = pd.DataFrame(samples)
        
        # Make predictions
        predictions = model.predict(df.values)
        
        return BatchPredictionOutput(
            predictions=[{{"q_factor": float(pred)}} for pred in predictions]
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {{
        "model_id": "{model_config['model_id']}",
        "version": "{model_config['version']}",
        "model_type": "{model_config['model_type']}",
        "framework": "{model_config['framework']}",
        "created_at": "{model_config['created_at']}"
    }}
'''
        
        with open(api_path, 'w') as f:
            f.write(api_content)
    
    def _get_dependencies(self) -> List[str]:
        """Get list of dependencies."""
        return [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scikit-learn>=1.0.0",
            "joblib>=1.0.0",
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0"
        ]
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate MD5 checksum of directory contents."""
        hash_md5 = hashlib.md5()
        
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in sorted(filenames):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _create_archive(self, source_dir: Path, archive_path: Path):
        """Create ZIP archive of the package."""
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arc_name)


class ContainerDeployer:
    """
    Deploy models using containerization.
    
    Supports Docker and Kubernetes deployments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize container deployer.
        
        Args:
            config: Deployment configuration.
        """
        self.config = config or {}
        self.docker_client = None
        self.k8s_client = None
        
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {e}")
        
        if HAS_KUBERNETES:
            try:
                config.load_incluster_config()
                self.k8s_client = client.ApiClient()
            except:
                try:
                    config.load_kube_config()
                    self.k8s_client = client.ApiClient()
                except Exception as e:
                    logger.warning(f"Kubernetes client initialization failed: {e}")
        
        logger.info("ContainerDeployer initialized")
    
    def build_docker_image(self, 
                           package_path: str,
                           image_name: str,
                           image_tag: str = "latest") -> str:
        """
        Build Docker image from model package.
        
        Args:
            package_path: Path to model package directory.
            image_name: Docker image name.
            image_tag: Docker image tag.
            
        Returns:
            Full image name with tag.
        """
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        full_image_name = f"{image_name}:{image_tag}"
        
        try:
            # Build image
            image, logs = self.docker_client.images.build(
                path=package_path,
                tag=full_image_name,
                rm=True
            )
            
            # Log build output
            for log in logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            logger.info(f"Docker image built successfully: {full_image_name}")
            return full_image_name
            
        except Exception as e:
            logger.error(f"Docker image build failed: {e}")
            raise
    
    def deploy_to_kubernetes(self, 
                             image_name: str,
                             deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy model to Kubernetes cluster.
        
        Args:
            image_name: Docker image name.
            deployment_config: Deployment configuration.
            
        Returns:
            Deployment status.
        """
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        apps_v1 = client.AppsV1Api(self.k8s_client)
        core_v1 = client.CoreV1Api(self.k8s_client)
        
        namespace = deployment_config.scaling_config.get('namespace', 'default')
        app_name = deployment_config.scaling_config.get('app_name', 'fusion-analyzer')
        
        # Create deployment
        deployment = self._create_k8s_deployment(image_name, deployment_config)
        
        try:
            # Apply deployment
            apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            # Create service
            service = self._create_k8s_service(deployment_config)
            core_v1.create_namespaced_service(
                namespace=namespace,
                body=service
            )
            
            logger.info(f"Kubernetes deployment created: {app_name}")
            
            return {
                'status': 'deployed',
                'namespace': namespace,
                'app_name': app_name,
                'image': image_name
            }
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    def _create_k8s_deployment(self, image_name: str, config: DeploymentConfig) -> Dict:
        """Create Kubernetes deployment manifest."""
        app_name = config.scaling_config.get('app_name', 'fusion-analyzer')
        replicas = config.scaling_config.get('replicas', 3)
        
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': app_name,
                'labels': {'app': app_name}
            },
            'spec': {
                'replicas': replicas,
                'selector': {'matchLabels': {'app': app_name}},
                'template': {
                    'metadata': {'labels': {'app': app_name}},
                    'spec': {
                        'containers': [{
                            'name': app_name,
                            'image': image_name,
                            'ports': [{'containerPort': 8000}],
                            'resources': config.resource_limits,
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return deployment
    
    def _create_k8s_service(self, config: DeploymentConfig) -> Dict:
        """Create Kubernetes service manifest."""
        app_name = config.scaling_config.get('app_name', 'fusion-analyzer')
        service_type = config.scaling_config.get('service_type', 'ClusterIP')
        
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{app_name}-service",
                'labels': {'app': app_name}
            },
            'spec': {
                'selector': {'app': app_name},
                'ports': [{'port': 80, 'targetPort': 8000}],
                'type': service_type
            }
        }
        
        return service


class CloudDeployer:
    """
    Deploy models to cloud platforms.
    
    Supports AWS SageMaker, Azure ML, and Google Cloud AI Platform.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cloud deployer.
        
        Args:
            config: Cloud deployment configuration.
        """
        self.config = config or {}
        self.aws_client = None
        
        if HAS_AWS:
            try:
                self.aws_client = boto3.client('sagemaker')
            except Exception as e:
                logger.warning(f"AWS client initialization failed: {e}")
        
        logger.info("CloudDeployer initialized")
    
    def deploy_to_sagemaker(self, 
                            model_package: ModelPackage,
                            deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy model to AWS SageMaker.
        
        Args:
            model_package: Model package information.
            deployment_config: Deployment configuration.
            
        Returns:
            Deployment status.
        """
        if not self.aws_client:
            raise RuntimeError("AWS SageMaker client not available")
        
        try:
            # Create model
            model_name = f"{model_package.model_id}-{model_package.version}"
            
            # Deploy to SageMaker endpoint
            endpoint_config_name = f"{model_name}-config"
            endpoint_name = f"{model_name}-endpoint"
            
            # Create endpoint configuration
            self.aws_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': deployment_config.scaling_config.get('instance_count', 1),
                    'InstanceType': deployment_config.scaling_config.get('instance_type', 'ml.t2.medium')
                }]
            )
            
            # Create endpoint
            self.aws_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"SageMaker deployment initiated: {endpoint_name}")
            
            return {
                'status': 'deploying',
                'endpoint_name': endpoint_name,
                'model_name': model_name
            }
            
        except ClientError as e:
            logger.error(f"SageMaker deployment failed: {e}")
            raise


def create_deployment_system(config_path: Optional[str] = None) -> Tuple[ModelPackager, ContainerDeployer, CloudDeployer]:
    """
    Create complete deployment system.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Tuple of (packager, container_deployer, cloud_deployer).
    """
    config = {}
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config().get('deployment', {})
    
    packager = ModelPackager(config.get('packaging', {}))
    container_deployer = ContainerDeployer(config.get('container', {}))
    cloud_deployer = CloudDeployer(config.get('cloud', {}))
    
    return packager, container_deployer, cloud_deployer