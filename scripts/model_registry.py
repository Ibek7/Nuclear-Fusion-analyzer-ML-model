#!/usr/bin/env python3
"""
Model versioning and registry management for Nuclear Fusion Analyzer.

This script provides comprehensive model versioning capabilities including
model registration, version management, metadata tracking, model comparison,
rollback functionality, and A/B testing support.

Usage:
    python model_registry.py register --model-path ./models/fusion_model.joblib --version 1.0.0
    python model_registry.py list --status active
    python model_registry.py compare --version1 1.0.0 --version2 1.1.0
    python model_registry.py rollback --version 1.0.0
"""

import argparse
import json
import sqlite3
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from src.models.fusion_predictor import FusionPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata container."""
    
    model_id: str
    version: str
    name: str
    description: str
    author: str
    created_at: str
    model_type: str
    framework: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    performance_metrics: Dict[str, float]
    file_path: str
    file_size: int
    file_hash: str
    status: str  # active, deprecated, archived
    tags: List[str]
    parent_version: Optional[str]
    deployment_config: Dict[str, Any]


class ModelRegistry:
    """
    Comprehensive model registry for version management.
    
    Provides model registration, versioning, metadata tracking,
    comparison, rollback, and A/B testing capabilities.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry directory.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.registry_path / "registry.db"
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.registry_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self._init_database()
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT,
                    version TEXT,
                    name TEXT,
                    description TEXT,
                    author TEXT,
                    created_at TEXT,
                    model_type TEXT,
                    framework TEXT,
                    algorithm TEXT,
                    hyperparameters TEXT,
                    training_data_hash TEXT,
                    performance_metrics TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    file_hash TEXT,
                    status TEXT,
                    tags TEXT,
                    parent_version TEXT,
                    deployment_config TEXT,
                    PRIMARY KEY (model_id, version)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    models TEXT,
                    traffic_split TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    results TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_status ON models(status)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)
            ''')
    
    def register_model(self,
                      model_path: str,
                      model_id: str,
                      version: str,
                      name: str,
                      description: str = "",
                      author: str = "",
                      tags: List[str] = None,
                      parent_version: Optional[str] = None,
                      performance_metrics: Dict[str, float] = None,
                      deployment_config: Dict[str, Any] = None) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            model_path: Path to model file.
            model_id: Unique model identifier.
            version: Model version.
            name: Human-readable model name.
            description: Model description.
            author: Model author.
            tags: Model tags.
            parent_version: Parent version for lineage tracking.
            performance_metrics: Model performance metrics.
            deployment_config: Deployment configuration.
            
        Returns:
            ModelMetadata for registered model.
        """
        # Check if version already exists
        if self._version_exists(model_id, version):
            raise ValueError(f"Model {model_id} version {version} already exists")
        
        # Load and analyze model
        model = joblib.load(model_path)
        
        # Extract model information
        model_info = self._extract_model_info(model)
        
        # Calculate file hash and size
        file_hash = self._calculate_file_hash(model_path)
        file_size = Path(model_path).stat().st_size
        
        # Copy model to registry
        registry_model_path = self.models_path / f"{model_id}_{version}.joblib"
        shutil.copy2(model_path, registry_model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            author=author or "unknown",
            created_at=datetime.now().isoformat(),
            model_type=model_info['type'],
            framework=model_info['framework'],
            algorithm=model_info['algorithm'],
            hyperparameters=model_info['hyperparameters'],
            training_data_hash="",  # Would be provided separately
            performance_metrics=performance_metrics or {},
            file_path=str(registry_model_path),
            file_size=file_size,
            file_hash=file_hash,
            status="active",
            tags=tags or [],
            parent_version=parent_version,
            deployment_config=deployment_config or {}
        )
        
        # Save metadata to database
        self._save_metadata(metadata)
        
        # Save metadata JSON
        metadata_file = self.metadata_path / f"{model_id}_{version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Model registered: {model_id} v{version}")
        return metadata
    
    def list_models(self,
                   model_id: Optional[str] = None,
                   status: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List models in registry.
        
        Args:
            model_id: Filter by model ID.
            status: Filter by status.
            tags: Filter by tags.
            
        Returns:
            List of ModelMetadata.
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        models = []
        for row in rows:
            metadata = self._row_to_metadata(row)
            
            # Filter by tags if specified
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            models.append(metadata)
        
        return models
    
    def get_model(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """
        Get specific model version.
        
        Args:
            model_id: Model identifier.
            version: Model version.
            
        Returns:
            ModelMetadata if found, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM models WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
            row = cursor.fetchone()
        
        if row:
            return self._row_to_metadata(row)
        return None
    
    def get_latest_version(self, model_id: str, status: str = "active") -> Optional[ModelMetadata]:
        """
        Get latest version of a model.
        
        Args:
            model_id: Model identifier.
            status: Model status filter.
            
        Returns:
            Latest ModelMetadata if found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM models WHERE model_id = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
                (model_id, status)
            )
            row = cursor.fetchone()
        
        if row:
            return self._row_to_metadata(row)
        return None
    
    def update_status(self, model_id: str, version: str, status: str):
        """
        Update model status.
        
        Args:
            model_id: Model identifier.
            version: Model version.
            status: New status (active, deprecated, archived).
        """
        valid_statuses = ["active", "deprecated", "archived"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE models SET status = ? WHERE model_id = ? AND version = ?",
                (status, model_id, version)
            )
        
        logger.info(f"Updated {model_id} v{version} status to {status}")
    
    def compare_models(self, 
                      model_id: str,
                      version1: str,
                      version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_id: Model identifier.
            version1: First version to compare.
            version2: Second version to compare.
            
        Returns:
            Comparison results.
        """
        model1 = self.get_model(model_id, version1)
        model2 = self.get_model(model_id, version2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            'model_id': model_id,
            'versions': {
                'version1': version1,
                'version2': version2
            },
            'metadata_diff': self._compare_metadata(model1, model2),
            'performance_diff': self._compare_performance(model1, model2),
            'model_diff': self._compare_model_files(model1, model2)
        }
        
        return comparison
    
    def create_experiment(self,
                         experiment_id: str,
                         name: str,
                         description: str,
                         models: List[Tuple[str, str]],
                         traffic_split: Dict[str, float]) -> str:
        """
        Create A/B testing experiment.
        
        Args:
            experiment_id: Unique experiment identifier.
            name: Experiment name.
            description: Experiment description.
            models: List of (model_id, version) tuples.
            traffic_split: Traffic split percentages.
            
        Returns:
            Experiment ID.
        """
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 1e-6:
            raise ValueError("Traffic split must sum to 1.0")
        
        experiment_data = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'models': models,
            'traffic_split': traffic_split,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'active',
            'results': {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiments 
                (experiment_id, name, description, models, traffic_split, 
                 start_time, end_time, status, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, name, description,
                json.dumps(models), json.dumps(traffic_split),
                experiment_data['start_time'], None, 'active',
                json.dumps({})
            ))
        
        logger.info(f"Experiment created: {experiment_id}")
        return experiment_id
    
    def rollback_model(self, model_id: str, target_version: str):
        """
        Rollback to a previous model version.
        
        Args:
            model_id: Model identifier.
            target_version: Version to rollback to.
        """
        target_model = self.get_model(model_id, target_version)
        if not target_model:
            raise ValueError(f"Target version {target_version} not found")
        
        # Deactivate current active models
        current_models = self.list_models(model_id=model_id, status="active")
        for model in current_models:
            self.update_status(model_id, model.version, "deprecated")
        
        # Activate target version
        self.update_status(model_id, target_version, "active")
        
        logger.info(f"Rolled back {model_id} to version {target_version}")
    
    def delete_model(self, model_id: str, version: str):
        """
        Delete a model version.
        
        Args:
            model_id: Model identifier.
            version: Model version.
        """
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        # Delete model file
        if Path(model.file_path).exists():
            Path(model.file_path).unlink()
        
        # Delete metadata file
        metadata_file = self.metadata_path / f"{model_id}_{version}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM models WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
        
        logger.info(f"Deleted model {model_id} version {version}")
    
    def _version_exists(self, model_id: str, version: str) -> bool:
        """Check if model version exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM models WHERE model_id = ? AND version = ?",
                (model_id, version)
            )
            return cursor.fetchone() is not None
    
    def _extract_model_info(self, model) -> Dict[str, Any]:
        """Extract information from model object."""
        info = {
            'type': 'fusion_predictor',
            'framework': 'scikit-learn',
            'algorithm': 'ensemble',
            'hyperparameters': {}
        }
        
        if hasattr(model, 'models'):
            algorithms = list(model.models.keys())
            info['algorithm'] = ', '.join(algorithms)
            
            # Extract hyperparameters
            for name, estimator in model.models.items():
                if hasattr(estimator, 'get_params'):
                    info['hyperparameters'][name] = estimator.get_params()
        
        return info
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, version, name, description, author, created_at,
                 model_type, framework, algorithm, hyperparameters,
                 training_data_hash, performance_metrics, file_path,
                 file_size, file_hash, status, tags, parent_version,
                 deployment_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id, metadata.version, metadata.name,
                metadata.description, metadata.author, metadata.created_at,
                metadata.model_type, metadata.framework, metadata.algorithm,
                json.dumps(metadata.hyperparameters),
                metadata.training_data_hash,
                json.dumps(metadata.performance_metrics),
                metadata.file_path, metadata.file_size, metadata.file_hash,
                metadata.status, json.dumps(metadata.tags),
                metadata.parent_version,
                json.dumps(metadata.deployment_config)
            ))
    
    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row['model_id'],
            version=row['version'],
            name=row['name'],
            description=row['description'],
            author=row['author'],
            created_at=row['created_at'],
            model_type=row['model_type'],
            framework=row['framework'],
            algorithm=row['algorithm'],
            hyperparameters=json.loads(row['hyperparameters']),
            training_data_hash=row['training_data_hash'],
            performance_metrics=json.loads(row['performance_metrics']),
            file_path=row['file_path'],
            file_size=row['file_size'],
            file_hash=row['file_hash'],
            status=row['status'],
            tags=json.loads(row['tags']),
            parent_version=row['parent_version'],
            deployment_config=json.loads(row['deployment_config'])
        )
    
    def _compare_metadata(self, model1: ModelMetadata, model2: ModelMetadata) -> Dict:
        """Compare model metadata."""
        diff = {}
        
        for field in ['algorithm', 'framework', 'file_size', 'author']:
            val1 = getattr(model1, field)
            val2 = getattr(model2, field)
            if val1 != val2:
                diff[field] = {'version1': val1, 'version2': val2}
        
        return diff
    
    def _compare_performance(self, model1: ModelMetadata, model2: ModelMetadata) -> Dict:
        """Compare model performance metrics."""
        metrics1 = model1.performance_metrics
        metrics2 = model2.performance_metrics
        
        comparison = {}
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, None)
            val2 = metrics2.get(metric, None)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                comparison[metric] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': diff,
                    'improvement': diff > 0
                }
            elif val1 is not None:
                comparison[metric] = {'version1': val1, 'version2': None}
            elif val2 is not None:
                comparison[metric] = {'version1': None, 'version2': val2}
        
        return comparison
    
    def _compare_model_files(self, model1: ModelMetadata, model2: ModelMetadata) -> Dict:
        """Compare model files."""
        return {
            'file_size_diff': model2.file_size - model1.file_size,
            'hash_different': model1.file_hash != model2.file_hash,
            'version1_hash': model1.file_hash,
            'version2_hash': model2.file_hash
        }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Nuclear Fusion Model Registry")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new model')
    register_parser.add_argument('--model-path', required=True, help='Path to model file')
    register_parser.add_argument('--model-id', required=True, help='Model identifier')
    register_parser.add_argument('--version', required=True, help='Model version')
    register_parser.add_argument('--name', required=True, help='Model name')
    register_parser.add_argument('--description', default='', help='Model description')
    register_parser.add_argument('--author', default='', help='Model author')
    register_parser.add_argument('--tags', nargs='*', default=[], help='Model tags')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--model-id', help='Filter by model ID')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--tags', nargs='*', help='Filter by tags')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('--model-id', required=True, help='Model identifier')
    compare_parser.add_argument('--version1', required=True, help='First version')
    compare_parser.add_argument('--version2', required=True, help='Second version')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous version')
    rollback_parser.add_argument('--model-id', required=True, help='Model identifier')
    rollback_parser.add_argument('--version', required=True, help='Target version')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Update model status')
    status_parser.add_argument('--model-id', required=True, help='Model identifier')
    status_parser.add_argument('--version', required=True, help='Model version')
    status_parser.add_argument('--new-status', required=True, 
                              choices=['active', 'deprecated', 'archived'],
                              help='New status')
    
    parser.add_argument('--registry-path', default='./model_registry', 
                       help='Path to model registry')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize registry
    registry = ModelRegistry(args.registry_path)
    
    try:
        if args.command == 'register':
            metadata = registry.register_model(
                model_path=args.model_path,
                model_id=args.model_id,
                version=args.version,
                name=args.name,
                description=args.description,
                author=args.author,
                tags=args.tags
            )
            print(f"Model registered successfully: {args.model_id} v{args.version}")
            print(json.dumps(asdict(metadata), indent=2, default=str))
        
        elif args.command == 'list':
            models = registry.list_models(
                model_id=args.model_id,
                status=args.status,
                tags=args.tags
            )
            
            if models:
                print(f"Found {len(models)} models:")
                for model in models:
                    print(f"  {model.model_id} v{model.version} ({model.status}) - {model.name}")
            else:
                print("No models found")
        
        elif args.command == 'compare':
            comparison = registry.compare_models(
                model_id=args.model_id,
                version1=args.version1,
                version2=args.version2
            )
            print("Model comparison:")
            print(json.dumps(comparison, indent=2, default=str))
        
        elif args.command == 'rollback':
            registry.rollback_model(args.model_id, args.version)
            print(f"Rollback completed: {args.model_id} -> v{args.version}")
        
        elif args.command == 'status':
            registry.update_status(args.model_id, args.version, args.new_status)
            print(f"Status updated: {args.model_id} v{args.version} -> {args.new_status}")
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()