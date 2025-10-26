#!/usr/bin/env python3
"""
Model deployment automation script for Nuclear Fusion Analyzer.

This script provides command-line interface for deploying fusion models
to various platforms including containers, cloud services, and edge devices.

Usage:
    python deploy_model.py --model-path ./models/fusion_model.joblib --platform docker
    python deploy_model.py --config deployment_config.yaml --platform kubernetes
    python deploy_model.py --model-id fusion-v1.0 --platform aws-sagemaker
"""

import argparse
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fusion_predictor import FusionPredictor
from src.utils.deployment_utils import (
    ModelPackager, ContainerDeployer, CloudDeployer,
    ModelPackage, DeploymentConfig, create_deployment_system
)
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_model(model_path: str) -> FusionPredictor:
    """Load trained model from file."""
    try:
        import joblib
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_deployment_config(config_path: str) -> DeploymentConfig:
    """Load deployment configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        return DeploymentConfig(**config_data)
    except Exception as e:
        logger.error(f"Failed to load deployment config: {e}")
        raise


def create_default_deployment_config(platform: str) -> DeploymentConfig:
    """Create default deployment configuration for platform."""
    configs = {
        'docker': DeploymentConfig(
            environment='production',
            target_platform='docker',
            scaling_config={
                'replicas': 1,
                'port': 8000
            },
            resource_limits={
                'requests': {'memory': '1Gi', 'cpu': '500m'},
                'limits': {'memory': '2Gi', 'cpu': '1000m'}
            },
            health_check={
                'path': '/health',
                'interval': 30,
                'timeout': 5
            },
            monitoring={
                'enabled': True,
                'metrics_port': 9090
            },
            rollback_config={
                'enabled': True,
                'max_rollback_revisions': 3
            }
        ),
        'kubernetes': DeploymentConfig(
            environment='production',
            target_platform='kubernetes',
            scaling_config={
                'replicas': 3,
                'app_name': 'fusion-analyzer',
                'namespace': 'default',
                'service_type': 'LoadBalancer'
            },
            resource_limits={
                'requests': {'memory': '1Gi', 'cpu': '500m'},
                'limits': {'memory': '4Gi', 'cpu': '2000m'}
            },
            health_check={
                'path': '/health',
                'initial_delay': 30,
                'period': 10
            },
            monitoring={
                'enabled': True,
                'prometheus': True
            },
            rollback_config={
                'enabled': True,
                'strategy': 'RollingUpdate'
            }
        ),
        'aws-sagemaker': DeploymentConfig(
            environment='production',
            target_platform='aws-sagemaker',
            scaling_config={
                'instance_count': 1,
                'instance_type': 'ml.m5.large',
                'auto_scaling': True
            },
            resource_limits={
                'max_concurrent_invocations': 1000
            },
            health_check={
                'enabled': True
            },
            monitoring={
                'cloudwatch': True,
                'data_capture': True
            },
            rollback_config={
                'enabled': True,
                'traffic_routing': 'BlueGreen'
            }
        )
    }
    
    return configs.get(platform, configs['docker'])


def deploy_to_docker(packager: ModelPackager,
                     container_deployer: ContainerDeployer,
                     model: FusionPredictor,
                     model_id: str,
                     version: str,
                     config: DeploymentConfig) -> Dict[str, Any]:
    """Deploy model to Docker container."""
    logger.info("Starting Docker deployment...")
    
    # Package model
    package = packager.package_model(
        predictor=model,
        model_id=model_id,
        version=version,
        output_dir="./deployments"
    )
    
    # Build Docker image
    package_dir = f"./deployments/{model_id}_{version}"
    image_name = f"fusion-analyzer/{model_id}"
    
    full_image_name = container_deployer.build_docker_image(
        package_path=package_dir,
        image_name=image_name,
        image_tag=version
    )
    
    return {
        'status': 'success',
        'image_name': full_image_name,
        'package_info': package
    }


def deploy_to_kubernetes(packager: ModelPackager,
                        container_deployer: ContainerDeployer,
                        model: FusionPredictor,
                        model_id: str,
                        version: str,
                        config: DeploymentConfig) -> Dict[str, Any]:
    """Deploy model to Kubernetes cluster."""
    logger.info("Starting Kubernetes deployment...")
    
    # First create Docker image
    docker_result = deploy_to_docker(
        packager, container_deployer, model, model_id, version, config
    )
    
    # Deploy to Kubernetes
    k8s_result = container_deployer.deploy_to_kubernetes(
        image_name=docker_result['image_name'],
        deployment_config=config
    )
    
    return {
        'status': 'success',
        'kubernetes_deployment': k8s_result,
        'docker_image': docker_result['image_name']
    }


def deploy_to_aws_sagemaker(packager: ModelPackager,
                           cloud_deployer: CloudDeployer,
                           model: FusionPredictor,
                           model_id: str,
                           version: str,
                           config: DeploymentConfig) -> Dict[str, Any]:
    """Deploy model to AWS SageMaker."""
    logger.info("Starting AWS SageMaker deployment...")
    
    # Package model
    package = packager.package_model(
        predictor=model,
        model_id=model_id,
        version=version,
        output_dir="./deployments"
    )
    
    # Deploy to SageMaker
    sagemaker_result = cloud_deployer.deploy_to_sagemaker(
        model_package=package,
        deployment_config=config
    )
    
    return {
        'status': 'success',
        'sagemaker_deployment': sagemaker_result,
        'package_info': package
    }


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Nuclear Fusion Analyzer models"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='fusion-analyzer',
        help='Model identifier'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='1.0.0',
        help='Model version'
    )
    
    parser.add_argument(
        '--platform',
        type=str,
        choices=['docker', 'kubernetes', 'aws-sagemaker'],
        required=True,
        help='Deployment platform'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to deployment configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./deployments',
        help='Output directory for deployment artifacts'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load deployment configuration
        if args.config:
            deployment_config = load_deployment_config(args.config)
        else:
            deployment_config = create_default_deployment_config(args.platform)
        
        # Create deployment system
        packager, container_deployer, cloud_deployer = create_deployment_system()
        
        # Load model
        if args.model_path:
            model = load_model(args.model_path)
        else:
            # Create dummy model for demonstration
            from src.data.fusion_data_generator import FusionDataGenerator
            from src.models.fusion_predictor import FusionPredictor
            
            logger.info("Creating demo model for deployment...")
            generator = FusionDataGenerator()
            X, y = generator.generate_fusion_data(1000)
            
            model = FusionPredictor()
            model.fit(X, y)
        
        # Deploy based on platform
        if args.platform == 'docker':
            result = deploy_to_docker(
                packager, container_deployer, model,
                args.model_id, args.version, deployment_config
            )
        elif args.platform == 'kubernetes':
            result = deploy_to_kubernetes(
                packager, container_deployer, model,
                args.model_id, args.version, deployment_config
            )
        elif args.platform == 'aws-sagemaker':
            result = deploy_to_aws_sagemaker(
                packager, cloud_deployer, model,
                args.model_id, args.version, deployment_config
            )
        
        # Print results
        logger.info("Deployment completed successfully!")
        logger.info(f"Results: {json.dumps(result, indent=2, default=str)}")
        
        # Save deployment manifest
        manifest_path = Path(args.output_dir) / f"deployment_manifest_{args.model_id}_{args.version}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w') as f:
            json.dump({
                'deployment_info': result,
                'config': deployment_config.__dict__,
                'model_id': args.model_id,
                'version': args.version,
                'platform': args.platform,
                'timestamp': str(pd.Timestamp.now())
            }, f, indent=2, default=str)
        
        logger.info(f"Deployment manifest saved: {manifest_path}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()