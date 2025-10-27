"""
Database migration and schema management utilities.

This module provides database migration capabilities for schema updates,
data migrations, and version management across PostgreSQL and MongoDB.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass

try:
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import AsyncSession
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from alembic.runtime.migration import MigrationContext
    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False

from .models import Base
from .connection import DatabaseOrchestrator, DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class MigrationRecord:
    """Migration record for tracking applied migrations."""
    
    version: str
    name: str
    applied_at: datetime
    description: str
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class PostgreSQLMigrator:
    """
    PostgreSQL schema migration manager using Alembic.
    
    Handles schema versioning, migrations, and rollbacks
    for the PostgreSQL database.
    """
    
    def __init__(self, database_url: str, migrations_dir: str = "migrations"):
        """
        Initialize PostgreSQL migrator.
        
        Args:
            database_url: Database connection URL.
            migrations_dir: Directory containing migration files.
        """
        self.database_url = database_url
        self.migrations_dir = Path(migrations_dir)
        self.alembic_cfg = None
        
        if not HAS_ALEMBIC:
            logger.warning("Alembic not available, PostgreSQL migrations disabled")
        
        logger.info("PostgreSQLMigrator initialized")
    
    def initialize_alembic(self):
        """Initialize Alembic configuration."""
        if not HAS_ALEMBIC:
            raise RuntimeError("Alembic not installed")
        
        # Create migrations directory
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Create alembic.ini
        alembic_ini = self.migrations_dir / "alembic.ini"
        if not alembic_ini.exists():
            ini_content = f"""[alembic]
script_location = {self.migrations_dir}
sqlalchemy.url = {self.database_url}

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
            alembic_ini.write_text(ini_content)
        
        # Initialize alembic configuration
        self.alembic_cfg = Config(str(alembic_ini))
        self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
        
        # Initialize alembic environment if not exists
        env_py = self.migrations_dir / "env.py"
        if not env_py.exists():
            command.init(self.alembic_cfg, str(self.migrations_dir))
        
        logger.info("Alembic configuration initialized")
    
    def create_migration(self, message: str) -> str:
        """
        Create a new migration file.
        
        Args:
            message: Migration description.
            
        Returns:
            Migration file path.
        """
        if not self.alembic_cfg:
            self.initialize_alembic()
        
        # Generate migration
        command.revision(self.alembic_cfg, message=message, autogenerate=True)
        
        # Find the latest migration file
        versions_dir = self.migrations_dir / "versions"
        if versions_dir.exists():
            migration_files = list(versions_dir.glob("*.py"))
            if migration_files:
                latest_file = max(migration_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Created migration: {latest_file.name}")
                return str(latest_file)
        
        raise RuntimeError("Failed to create migration file")
    
    def apply_migrations(self) -> List[str]:
        """
        Apply pending migrations.
        
        Returns:
            List of applied migration versions.
        """
        if not self.alembic_cfg:
            self.initialize_alembic()
        
        # Get current and head revisions
        script = ScriptDirectory.from_config(self.alembic_cfg)
        
        # Apply migrations
        command.upgrade(self.alembic_cfg, "head")
        
        logger.info("PostgreSQL migrations applied")
        return []  # Would return actual versions in practice
    
    def rollback_migration(self, target_revision: str) -> bool:
        """
        Rollback to specific migration.
        
        Args:
            target_revision: Target migration revision.
            
        Returns:
            Success status.
        """
        if not self.alembic_cfg:
            self.initialize_alembic()
        
        try:
            command.downgrade(self.alembic_cfg, target_revision)
            logger.info(f"Rolled back to revision: {target_revision}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


class MongoDBMigrator:
    """
    MongoDB migration manager for schema and data migrations.
    
    Handles collection schema changes, index management,
    and data transformations.
    """
    
    def __init__(self, database_orchestrator: DatabaseOrchestrator):
        """
        Initialize MongoDB migrator.
        
        Args:
            database_orchestrator: Database orchestrator instance.
        """
        self.orchestrator = database_orchestrator
        self.migrations: List[Callable] = []
        
        logger.info("MongoDBMigrator initialized")
    
    def register_migration(self, version: str, description: str):
        """
        Decorator to register a migration function.
        
        Args:
            version: Migration version.
            description: Migration description.
        """
        def decorator(func: Callable):
            func._migration_version = version
            func._migration_description = description
            self.migrations.append(func)
            return func
        
        return decorator
    
    async def apply_migrations(self) -> List[MigrationRecord]:
        """
        Apply all registered migrations.
        
        Returns:
            List of migration records.
        """
        applied_migrations = []
        
        # Get migration history
        migration_history = await self._get_migration_history()
        applied_versions = {record['version'] for record in migration_history}
        
        # Sort migrations by version
        sorted_migrations = sorted(
            self.migrations, 
            key=lambda m: m._migration_version
        )
        
        for migration in sorted_migrations:
            version = migration._migration_version
            description = migration._migration_description
            
            if version in applied_versions:
                logger.info(f"Migration {version} already applied")
                continue
            
            start_time = datetime.now()
            
            try:
                # Execute migration
                await migration(self.orchestrator.mongodb.database)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Record successful migration
                record = MigrationRecord(
                    version=version,
                    name=migration.__name__,
                    applied_at=datetime.now(timezone.utc),
                    description=description,
                    success=True,
                    execution_time=execution_time
                )
                
                await self._record_migration(record)
                applied_migrations.append(record)
                
                logger.info(f"Applied MongoDB migration {version}: {description}")
                
            except Exception as e:
                # Record failed migration
                execution_time = (datetime.now() - start_time).total_seconds()
                
                record = MigrationRecord(
                    version=version,
                    name=migration.__name__,
                    applied_at=datetime.now(timezone.utc),
                    description=description,
                    success=False,
                    error_message=str(e),
                    execution_time=execution_time
                )
                
                await self._record_migration(record)
                applied_migrations.append(record)
                
                logger.error(f"Migration {version} failed: {e}")
                
                # Stop on first failure
                break
        
        return applied_migrations
    
    async def _get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history from database."""
        collection = self.orchestrator.mongodb.database.migration_history
        cursor = collection.find({}).sort("applied_at", 1)
        return await cursor.to_list(length=None)
    
    async def _record_migration(self, record: MigrationRecord):
        """Record migration in history."""
        collection = self.orchestrator.mongodb.database.migration_history
        
        document = {
            'version': record.version,
            'name': record.name,
            'applied_at': record.applied_at,
            'description': record.description,
            'success': record.success,
            'error_message': record.error_message,
            'execution_time': record.execution_time
        }
        
        await collection.insert_one(document)


class DataMigrator:
    """
    Data migration utilities for transforming and moving data.
    
    Handles data format changes, cleanup operations,
    and cross-database data synchronization.
    """
    
    def __init__(self, orchestrator: DatabaseOrchestrator):
        """
        Initialize data migrator.
        
        Args:
            orchestrator: Database orchestrator instance.
        """
        self.orchestrator = orchestrator
        
        logger.info("DataMigrator initialized")
    
    async def migrate_legacy_data(self, 
                                 source_format: str,
                                 target_format: str,
                                 batch_size: int = 1000) -> Dict[str, int]:
        """
        Migrate data from legacy format to new format.
        
        Args:
            source_format: Source data format identifier.
            target_format: Target data format identifier.
            batch_size: Processing batch size.
            
        Returns:
            Migration statistics.
        """
        stats = {
            'processed': 0,
            'migrated': 0,
            'errors': 0
        }
        
        if source_format == "legacy_v1" and target_format == "current_v2":
            stats = await self._migrate_v1_to_v2(batch_size)
        
        logger.info(f"Data migration completed: {stats}")
        return stats
    
    async def _migrate_v1_to_v2(self, batch_size: int) -> Dict[str, int]:
        """Migrate from v1 to v2 format."""
        stats = {'processed': 0, 'migrated': 0, 'errors': 0}
        
        # Get legacy data from MongoDB
        collection = self.orchestrator.mongodb.database.legacy_data
        cursor = collection.find({'format_version': 'v1'})
        
        batch = []
        async for document in cursor:
            stats['processed'] += 1
            
            try:
                # Transform document
                transformed = self._transform_v1_document(document)
                batch.append(transformed)
                
                # Process batch
                if len(batch) >= batch_size:
                    await self._process_migration_batch(batch)
                    stats['migrated'] += len(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error migrating document {document.get('_id')}: {e}")
                stats['errors'] += 1
        
        # Process remaining batch
        if batch:
            await self._process_migration_batch(batch)
            stats['migrated'] += len(batch)
        
        return stats
    
    def _transform_v1_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 document to v2 format."""
        transformed = {
            'id': str(document['_id']),
            'format_version': 'v2',
            'migrated_at': datetime.now(timezone.utc),
            'original_data': document,
            'data': {
                # Transform specific fields
                'experiment_id': document.get('exp_id'),
                'shot_number': document.get('shot_num'),
                'parameters': document.get('params', {}),
                'results': document.get('results', {})
            }
        }
        
        return transformed
    
    async def _process_migration_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of migrated documents."""
        # Store in new collection
        collection = self.orchestrator.mongodb.database.migrated_data_v2
        await collection.insert_many(batch)
    
    async def cleanup_orphaned_data(self) -> Dict[str, int]:
        """
        Clean up orphaned data across databases.
        
        Returns:
            Cleanup statistics.
        """
        stats = {
            'postgresql_orphans': 0,
            'mongodb_orphans': 0,
            'cache_cleared': 0
        }
        
        # Clean up orphaned time series data
        async with self.orchestrator.postgresql.get_session() as session:
            # Find time series without valid shots
            orphan_query = """
            DELETE FROM time_series_data 
            WHERE shot_id NOT IN (SELECT id FROM fusion_shots)
            """
            result = await session.execute(sa.text(orphan_query))
            stats['postgresql_orphans'] = result.rowcount
            await session.commit()
        
        # Clean up old MongoDB documents
        mongo_db = self.orchestrator.mongodb.database
        
        # Remove documents older than 1 year without references
        cutoff_date = datetime.now(timezone.utc).replace(year=datetime.now().year - 1)
        
        old_docs_query = {
            'timestamp': {'$lt': cutoff_date},
            'referenced': {'$ne': True}
        }
        
        result = await mongo_db.temporary_data.delete_many(old_docs_query)
        stats['mongodb_orphans'] = result.deleted_count
        
        # Clear expired cache entries
        if self.orchestrator.cache.redis_client:
            # Note: Redis automatically expires keys, but we can force cleanup
            stats['cache_cleared'] = 0  # Would implement actual cache cleanup
        
        logger.info(f"Cleanup completed: {stats}")
        return stats


class SchemaValidator:
    """
    Schema validation utilities for ensuring data integrity.
    
    Validates database schemas, checks constraints,
    and reports inconsistencies.
    """
    
    def __init__(self, orchestrator: DatabaseOrchestrator):
        """
        Initialize schema validator.
        
        Args:
            orchestrator: Database orchestrator instance.
        """
        self.orchestrator = orchestrator
        
        logger.info("SchemaValidator initialized")
    
    async def validate_postgresql_schema(self) -> Dict[str, Any]:
        """
        Validate PostgreSQL schema integrity.
        
        Returns:
            Validation report.
        """
        report = {
            'tables_validated': 0,
            'constraints_checked': 0,
            'indexes_verified': 0,
            'issues': []
        }
        
        async with self.orchestrator.postgresql.get_session() as session:
            # Check table existence
            tables_query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            
            result = await session.execute(sa.text(tables_query))
            tables = [row[0] for row in result]
            
            expected_tables = [
                'fusion_experiments', 
                'fusion_shots', 
                'time_series_data',
                'model_predictions'
            ]
            
            for table in expected_tables:
                if table in tables:
                    report['tables_validated'] += 1
                else:
                    report['issues'].append(f"Missing table: {table}")
            
            # Check constraints
            constraints_query = """
            SELECT constraint_name, table_name, constraint_type
            FROM information_schema.table_constraints
            WHERE table_schema = 'public'
            """
            
            result = await session.execute(sa.text(constraints_query))
            constraints = result.fetchall()
            report['constraints_checked'] = len(constraints)
            
            # Check indexes
            indexes_query = """
            SELECT indexname, tablename FROM pg_indexes
            WHERE schemaname = 'public'
            """
            
            result = await session.execute(sa.text(indexes_query))
            indexes = result.fetchall()
            report['indexes_verified'] = len(indexes)
        
        logger.info(f"PostgreSQL schema validation: {report}")
        return report
    
    async def validate_mongodb_collections(self) -> Dict[str, Any]:
        """
        Validate MongoDB collections and indexes.
        
        Returns:
            Validation report.
        """
        report = {
            'collections_validated': 0,
            'indexes_checked': 0,
            'issues': []
        }
        
        database = self.orchestrator.mongodb.database
        
        # Get collection names
        collections = await database.list_collection_names()
        
        expected_collections = [
            'raw_fusion_data',
            'analysis_results', 
            'model_artifacts',
            'migration_history'
        ]
        
        for collection_name in expected_collections:
            if collection_name in collections:
                report['collections_validated'] += 1
                
                # Check indexes
                collection = database[collection_name]
                indexes = await collection.list_indexes().to_list(length=None)
                report['indexes_checked'] += len(indexes)
                
            else:
                report['issues'].append(f"Missing collection: {collection_name}")
        
        logger.info(f"MongoDB validation: {report}")
        return report
    
    async def check_data_consistency(self) -> Dict[str, Any]:
        """
        Check data consistency across databases.
        
        Returns:
            Consistency report.
        """
        report = {
            'experiments_checked': 0,
            'consistency_issues': [],
            'orphaned_records': 0
        }
        
        # Get experiments from PostgreSQL
        experiments = await self.orchestrator.postgresql.get_experiments(limit=100)
        
        for experiment in experiments:
            report['experiments_checked'] += 1
            
            # Check if MongoDB has corresponding data
            mongo_data = await self.orchestrator.mongodb.get_raw_data(
                experiment_id=str(experiment.id),
                data_type='experimental_data'
            )
            
            if not mongo_data:
                report['consistency_issues'].append(
                    f"Experiment {experiment.id} missing MongoDB data"
                )
        
        logger.info(f"Data consistency check: {report}")
        return report


def create_migration_manager(orchestrator: DatabaseOrchestrator) -> Tuple[PostgreSQLMigrator, MongoDBMigrator, DataMigrator]:
    """
    Create migration managers for all databases.
    
    Args:
        orchestrator: Database orchestrator instance.
        
    Returns:
        Tuple of migration managers.
    """
    # Build PostgreSQL URL
    config = orchestrator.config
    postgresql_url = (
        f"postgresql+asyncpg://{config.postgresql_username}:"
        f"{config.postgresql_password}@{config.postgresql_host}:"
        f"{config.postgresql_port}/{config.postgresql_database}"
    )
    
    pg_migrator = PostgreSQLMigrator(postgresql_url)
    mongo_migrator = MongoDBMigrator(orchestrator)
    data_migrator = DataMigrator(orchestrator)
    
    return pg_migrator, mongo_migrator, data_migrator