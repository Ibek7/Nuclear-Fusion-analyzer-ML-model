"""
Database connection and configuration management.

This module provides centralized database connection management,
configuration handling, and connection pooling for PostgreSQL,
MongoDB, and Redis databases.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration container."""
    
    # PostgreSQL settings
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_database: str = "fusion_analyzer"
    postgresql_username: str = "fusion_user"
    postgresql_password: str = "fusion_password"
    postgresql_pool_size: int = 20
    postgresql_max_overflow: int = 40
    
    # MongoDB settings
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "fusion_analyzer"
    mongodb_username: Optional[str] = None
    mongodb_password: Optional[str] = None
    mongodb_auth_source: str = "admin"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Connection settings
    connection_timeout: int = 30
    retry_attempts: int = 3
    enable_ssl: bool = False
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        return cls(
            postgresql_host=os.getenv('POSTGRESQL_HOST', 'localhost'),
            postgresql_port=int(os.getenv('POSTGRESQL_PORT', '5432')),
            postgresql_database=os.getenv('POSTGRESQL_DATABASE', 'fusion_analyzer'),
            postgresql_username=os.getenv('POSTGRESQL_USERNAME', 'fusion_user'),
            postgresql_password=os.getenv('POSTGRESQL_PASSWORD', 'fusion_password'),
            mongodb_host=os.getenv('MONGODB_HOST', 'localhost'),
            mongodb_port=int(os.getenv('MONGODB_PORT', '27017')),
            mongodb_database=os.getenv('MONGODB_DATABASE', 'fusion_analyzer'),
            mongodb_username=os.getenv('MONGODB_USERNAME'),
            mongodb_password=os.getenv('MONGODB_PASSWORD'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_password=os.getenv('REDIS_PASSWORD'),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DatabaseConfig':
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data.get('database', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {'database': self.to_dict()}
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)


class ConnectionPool:
    """
    Generic connection pool implementation.
    
    Provides connection pooling functionality with
    health checks, retry logic, and automatic recovery.
    """
    
    def __init__(self, 
                 name: str,
                 min_connections: int = 5,
                 max_connections: int = 20,
                 health_check_interval: int = 60):
        """
        Initialize connection pool.
        
        Args:
            name: Pool name for logging.
            min_connections: Minimum connections to maintain.
            max_connections: Maximum connections allowed.
            health_check_interval: Health check interval in seconds.
        """
        self.name = name
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        
        self.active_connections = 0
        self.total_connections = 0
        self.failed_connections = 0
        self.last_health_check = datetime.now()
        
        self._pool_lock = asyncio.Lock()
        self._connections = asyncio.Queue(maxsize=max_connections)
        self._health_check_task = None
        
        logger.info(f"ConnectionPool '{name}' initialized")
    
    async def start(self):
        """Start connection pool and health monitoring."""
        # Start health check task
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
        
        logger.info(f"ConnectionPool '{self.name}' started")
    
    async def stop(self):
        """Stop connection pool and cleanup."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        while not self._connections.empty():
            try:
                connection = self._connections.get_nowait()
                await self._close_connection(connection)
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"ConnectionPool '{self.name}' stopped")
    
    async def get_connection(self):
        """Get connection from pool."""
        try:
            # Try to get existing connection
            connection = self._connections.get_nowait()
            
            # Validate connection
            if await self._validate_connection(connection):
                self.active_connections += 1
                return connection
            else:
                await self._close_connection(connection)
                
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if under limit
        async with self._pool_lock:
            if self.total_connections < self.max_connections:
                connection = await self._create_connection()
                if connection:
                    self.total_connections += 1
                    self.active_connections += 1
                    return connection
        
        # Wait for available connection
        connection = await self._connections.get()
        self.active_connections += 1
        return connection
    
    async def return_connection(self, connection):
        """Return connection to pool."""
        self.active_connections -= 1
        
        if await self._validate_connection(connection):
            try:
                self._connections.put_nowait(connection)
            except asyncio.QueueFull:
                await self._close_connection(connection)
                self.total_connections -= 1
        else:
            await self._close_connection(connection)
            self.total_connections -= 1
    
    async def _create_connection(self):
        """Create new connection (to be implemented by subclasses)."""
        raise NotImplementedError()
    
    async def _close_connection(self, connection):
        """Close connection (to be implemented by subclasses)."""
        raise NotImplementedError()
    
    async def _validate_connection(self, connection) -> bool:
        """Validate connection health (to be implemented by subclasses)."""
        raise NotImplementedError()
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error in pool '{self.name}': {e}")
    
    async def _perform_health_check(self):
        """Perform health check on pool."""
        self.last_health_check = datetime.now()
        
        # Check if we need to maintain minimum connections
        async with self._pool_lock:
            current_size = self._connections.qsize()
            needed = max(0, self.min_connections - current_size)
            
            for _ in range(needed):
                if self.total_connections < self.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        self.total_connections += 1
                        self._connections.put_nowait(connection)
        
        logger.debug(f"Health check completed for pool '{self.name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'name': self.name,
            'active_connections': self.active_connections,
            'total_connections': self.total_connections,
            'queue_size': self._connections.qsize(),
            'failed_connections': self.failed_connections,
            'last_health_check': self.last_health_check.isoformat(),
            'max_connections': self.max_connections,
            'min_connections': self.min_connections
        }


class ConnectionManager:
    """
    Centralized connection manager for all database systems.
    
    Coordinates connections across PostgreSQL, MongoDB, and Redis
    with unified configuration and monitoring.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize connection manager.
        
        Args:
            config: Database configuration.
        """
        self.config = config
        self.pools: Dict[str, ConnectionPool] = {}
        self.connections_active = False
        
        logger.info("ConnectionManager initialized")
    
    async def initialize(self):
        """Initialize all database connections."""
        try:
            # Initialize PostgreSQL pool
            await self._initialize_postgresql()
            
            # Initialize MongoDB pool
            await self._initialize_mongodb()
            
            # Initialize Redis pool
            await self._initialize_redis()
            
            # Start all pools
            for pool in self.pools.values():
                await pool.start()
            
            self.connections_active = True
            logger.info("All database connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            await self.close()
            raise
    
    async def close(self):
        """Close all database connections."""
        for pool in self.pools.values():
            await pool.stop()
        
        self.pools.clear()
        self.connections_active = False
        
        logger.info("All database connections closed")
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool."""
        try:
            # Check if PostgreSQL dependencies are available
            import asyncpg
            
            class PostgreSQLPool(ConnectionPool):
                def __init__(self, config: DatabaseConfig):
                    super().__init__(
                        name="postgresql",
                        min_connections=config.postgresql_pool_size // 4,
                        max_connections=config.postgresql_pool_size
                    )
                    self.config = config
                
                async def _create_connection(self):
                    try:
                        connection = await asyncpg.connect(
                            host=self.config.postgresql_host,
                            port=self.config.postgresql_port,
                            database=self.config.postgresql_database,
                            user=self.config.postgresql_username,
                            password=self.config.postgresql_password,
                            timeout=self.config.connection_timeout
                        )
                        return connection
                    except Exception as e:
                        self.failed_connections += 1
                        logger.error(f"Failed to create PostgreSQL connection: {e}")
                        return None
                
                async def _close_connection(self, connection):
                    try:
                        await connection.close()
                    except Exception as e:
                        logger.error(f"Error closing PostgreSQL connection: {e}")
                
                async def _validate_connection(self, connection) -> bool:
                    try:
                        await connection.fetchval('SELECT 1')
                        return True
                    except Exception:
                        return False
            
            self.pools['postgresql'] = PostgreSQLPool(self.config)
            logger.info("PostgreSQL pool initialized")
            
        except ImportError:
            logger.warning("PostgreSQL dependencies not available")
    
    async def _initialize_mongodb(self):
        """Initialize MongoDB connection pool."""
        try:
            # Check if MongoDB dependencies are available
            import motor.motor_asyncio
            
            class MongoDBPool(ConnectionPool):
                def __init__(self, config: DatabaseConfig):
                    super().__init__(
                        name="mongodb",
                        min_connections=5,
                        max_connections=20
                    )
                    self.config = config
                    self.client = None
                
                async def _create_connection(self):
                    try:
                        if not self.client:
                            # Build connection URL
                            if self.config.mongodb_username and self.config.mongodb_password:
                                connection_url = (
                                    f"mongodb://{self.config.mongodb_username}:"
                                    f"{self.config.mongodb_password}@{self.config.mongodb_host}:"
                                    f"{self.config.mongodb_port}/{self.config.mongodb_database}"
                                    f"?authSource={self.config.mongodb_auth_source}"
                                )
                            else:
                                connection_url = (
                                    f"mongodb://{self.config.mongodb_host}:"
                                    f"{self.config.mongodb_port}"
                                )
                            
                            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                                connection_url,
                                serverSelectionTimeoutMS=self.config.connection_timeout * 1000
                            )
                        
                        # Return database instance
                        return self.client[self.config.mongodb_database]
                    
                    except Exception as e:
                        self.failed_connections += 1
                        logger.error(f"Failed to create MongoDB connection: {e}")
                        return None
                
                async def _close_connection(self, connection):
                    # MongoDB connections are managed by the client
                    pass
                
                async def _validate_connection(self, connection) -> bool:
                    try:
                        await connection.command('ping')
                        return True
                    except Exception:
                        return False
            
            self.pools['mongodb'] = MongoDBPool(self.config)
            logger.info("MongoDB pool initialized")
            
        except ImportError:
            logger.warning("MongoDB dependencies not available")
    
    async def _initialize_redis(self):
        """Initialize Redis connection pool."""
        try:
            # Check if Redis dependencies are available
            import redis.asyncio as redis
            
            class RedisPool(ConnectionPool):
                def __init__(self, config: DatabaseConfig):
                    super().__init__(
                        name="redis",
                        min_connections=3,
                        max_connections=10
                    )
                    self.config = config
                
                async def _create_connection(self):
                    try:
                        connection = redis.Redis(
                            host=self.config.redis_host,
                            port=self.config.redis_port,
                            db=self.config.redis_db,
                            password=self.config.redis_password,
                            decode_responses=True,
                            socket_timeout=self.config.connection_timeout
                        )
                        
                        # Test connection
                        await connection.ping()
                        return connection
                    
                    except Exception as e:
                        self.failed_connections += 1
                        logger.error(f"Failed to create Redis connection: {e}")
                        return None
                
                async def _close_connection(self, connection):
                    try:
                        await connection.close()
                    except Exception as e:
                        logger.error(f"Error closing Redis connection: {e}")
                
                async def _validate_connection(self, connection) -> bool:
                    try:
                        await connection.ping()
                        return True
                    except Exception:
                        return False
            
            self.pools['redis'] = RedisPool(self.config)
            logger.info("Redis pool initialized")
            
        except ImportError:
            logger.warning("Redis dependencies not available")
    
    @asynccontextmanager
    async def get_postgresql_connection(self):
        """Get PostgreSQL connection from pool."""
        if 'postgresql' not in self.pools:
            raise RuntimeError("PostgreSQL pool not available")
        
        pool = self.pools['postgresql']
        connection = await pool.get_connection()
        
        try:
            yield connection
        finally:
            await pool.return_connection(connection)
    
    @asynccontextmanager
    async def get_mongodb_connection(self):
        """Get MongoDB connection from pool."""
        if 'mongodb' not in self.pools:
            raise RuntimeError("MongoDB pool not available")
        
        pool = self.pools['mongodb']
        connection = await pool.get_connection()
        
        try:
            yield connection
        finally:
            await pool.return_connection(connection)
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """Get Redis connection from pool."""
        if 'redis' not in self.pools:
            raise RuntimeError("Redis pool not available")
        
        pool = self.pools['redis']
        connection = await pool.get_connection()
        
        try:
            yield connection
        finally:
            await pool.return_connection(connection)
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connection pools."""
        return {
            name: pool.get_stats() 
            for name, pool in self.pools.items()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all connections."""
        results = {}
        
        for name, pool in self.pools.items():
            try:
                # Get a connection and test it
                connection = await pool.get_connection()
                healthy = await pool._validate_connection(connection)
                await pool.return_connection(connection)
                results[name] = healthy
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
        
        return results


# Singleton instance for global access
_connection_manager = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        raise RuntimeError("Connection manager not initialized")
    return _connection_manager


def initialize_connection_manager(config: DatabaseConfig) -> ConnectionManager:
    """Initialize the global connection manager."""
    global _connection_manager
    _connection_manager = ConnectionManager(config)
    return _connection_manager


async def close_connection_manager():
    """Close the global connection manager."""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.close()
        _connection_manager = None