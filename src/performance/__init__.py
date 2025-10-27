"""
Advanced Caching and Performance Optimization System.

This module provides:
- Multi-layer caching with Redis and in-memory strategies
- Performance profiling and monitoring
- Memory optimization and garbage collection tuning
- Query optimization and result caching
- Distributed caching across microservices
- Cache invalidation and consistency management
- Performance metrics and benchmarking
"""

import asyncio
import json
import time
import pickle
import hashlib
import psutil
import gc
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone, timedelta
from functools import wraps
import logging
import threading
from collections import OrderedDict, defaultdict

# Redis for distributed caching
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# Memory profiling
try:
    import psutil
    import tracemalloc
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

# Performance monitoring
try:
    import py_spy
    HAS_PY_SPY = False  # Usually not available in runtime
except ImportError:
    HAS_PY_SPY = False

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    RANDOM = "random"


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"  # In-process memory
    L2_REDIS = "l2_redis"    # Redis distributed cache
    L3_DISK = "l3_disk"      # Disk-based cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live in seconds
    size: int = 0  # Size in bytes
    
    def __post_init__(self):
        """Calculate entry size."""
        try:
            self.size = len(pickle.dumps(self.value))
        except Exception:
            self.size = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class Cache(ABC):
    """Abstract cache interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(Cache):
    """In-memory cache implementation with various strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100 * 1024 * 1024,  # 100MB
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries.
            max_memory: Maximum memory usage in bytes.
            strategy: Cache replacement strategy.
            default_ttl: Default time to live in seconds.
        """
        self.max_size = max_size
        self.max_memory = max_memory
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.access_frequency: defaultdict = defaultdict(int)  # For LFU
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"MemoryCache initialized with strategy {strategy.value}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self.lock:
            if key not in self.entries:
                self.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired():
                del self.entries[key]
                self.access_order.pop(key, None)
                self.misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            self.hits += 1
            
            # Update access tracking for cache strategies
            if self.strategy == CacheStrategy.LRU:
                self.access_order.move_to_end(key)
            elif self.strategy == CacheStrategy.LFU:
                self.access_frequency[key] += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self.lock:
            try:
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl or self.default_ttl
                )
                
                # Check memory limit
                current_memory = self._get_current_memory()
                if current_memory + entry.size > self.max_memory:
                    await self._evict_entries(entry.size)
                
                # Check size limit
                if len(self.entries) >= self.max_size:
                    await self._evict_one()
                
                # Store entry
                self.entries[key] = entry
                
                # Update access tracking
                if self.strategy == CacheStrategy.LRU:
                    self.access_order[key] = True
                elif self.strategy == CacheStrategy.LFU:
                    self.access_frequency[key] = 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache entry: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self.lock:
            if key in self.entries:
                del self.entries[key]
                self.access_order.pop(key, None)
                self.access_frequency.pop(key, None)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.lock:
            if key not in self.entries:
                return False
            
            entry = self.entries[key]
            if entry.is_expired():
                del self.entries[key]
                self.access_order.pop(key, None)
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
            
            return {
                "entries": len(self.entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "memory_usage": self._get_current_memory(),
                "max_memory": self.max_memory,
                "strategy": self.strategy.value
            }
    
    def _get_current_memory(self) -> int:
        """Get current memory usage."""
        return sum(entry.size for entry in self.entries.values())
    
    async def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self.entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key = next(iter(self.access_order))
            del self.entries[key]
            del self.access_order[key]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_freq = min(self.access_frequency.values())
            for key, freq in self.access_frequency.items():
                if freq == min_freq:
                    del self.entries[key]
                    del self.access_frequency[key]
                    break
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].created_at)
            del self.entries[oldest_key]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, e in self.entries.items() if e.is_expired()]
            if expired_keys:
                del self.entries[expired_keys[0]]
            else:
                oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].created_at)
                del self.entries[oldest_key]
        
        elif self.strategy == CacheStrategy.RANDOM:
            # Remove random entry
            import random
            key = random.choice(list(self.entries.keys()))
            del self.entries[key]
        
        self.evictions += 1
    
    async def _evict_entries(self, required_space: int):
        """Evict entries to free up required space."""
        freed_space = 0
        
        while freed_space < required_space and self.entries:
            entry_size = next(iter(self.entries.values())).size
            await self._evict_one()
            freed_space += entry_size


class RedisCache(Cache):
    """Redis-based distributed cache."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "fusion:",
        default_ttl: Optional[int] = 3600,
        serializer: str = "pickle"
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for all cache keys.
            default_ttl: Default time to live in seconds.
            serializer: Serialization method (pickle, json).
        """
        if not HAS_REDIS:
            raise RuntimeError("Redis not available")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.serializer = serializer
        
        self.redis_client: Optional[redis.Redis] = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        logger.info("RedisCache initialized")
    
    async def connect(self):
        """Connect to Redis."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        logger.info("Connected to Redis cache")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Disconnected from Redis cache")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serializer == "pickle":
            return pickle.dumps(value)
        elif self.serializer == "json":
            return json.dumps(value).encode('utf-8')
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.serializer == "pickle":
            return pickle.loads(data)
        elif self.serializer == "json":
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                self.misses += 1
                return None
            
            value = self._deserialize(data)
            self.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            
            ttl = ttl or self.default_ttl
            if ttl:
                await self.redis_client.setex(redis_key, ttl, data)
            else:
                await self.redis_client.set(redis_key, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
            
            self.hits = 0
            self.misses = 0
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.exists(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error checking Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "type": "redis",
            "serializer": self.serializer
        }


class MultiLevelCache(Cache):
    """Multi-level cache hierarchy."""
    
    def __init__(self, caches: List[Tuple[CacheLevel, Cache]]):
        """
        Initialize multi-level cache.
        
        Args:
            caches: List of (level, cache) tuples in order of priority.
        """
        self.caches = sorted(caches, key=lambda x: x[0].value)
        
        logger.info(f"MultiLevelCache initialized with {len(self.caches)} levels")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        for level, cache in self.caches:
            value = await cache.get(key)
            if value is not None:
                # Propagate to higher levels
                await self._propagate_up(key, value, level)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache levels."""
        results = []
        
        for level, cache in self.caches:
            result = await cache.set(key, value, ttl)
            results.append(result)
        
        return any(results)
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        results = []
        
        for level, cache in self.caches:
            result = await cache.delete(key)
            results.append(result)
        
        return any(results)
    
    async def clear(self) -> bool:
        """Clear all cache levels."""
        results = []
        
        for level, cache in self.caches:
            result = await cache.clear()
            results.append(result)
        
        return all(results)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level."""
        for level, cache in self.caches:
            if await cache.exists(key):
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        stats = {}
        
        for level, cache in self.caches:
            stats[level.value] = cache.get_stats()
        
        return stats
    
    async def _propagate_up(self, key: str, value: Any, found_level: CacheLevel):
        """Propagate cache hit to higher levels."""
        for level, cache in self.caches:
            if level.value < found_level.value:
                await cache.set(key, value)
            elif level == found_level:
                break


def cache_result(
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None,
    cache_instance: Optional[Cache] = None
):
    """
    Decorator for caching function results.
    
    Args:
        key_func: Function to generate cache key.
        ttl: Time to live for cached result.
        cache_instance: Cache instance to use.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = cache_instance or getattr(wrapper, '_cache', None)
            if not cache:
                # Create default memory cache
                cache = MemoryCache()
                wrapper._cache = cache
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """Performance profiling and monitoring."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        
        if HAS_PROFILING:
            tracemalloc.start()
        
        logger.info("PerformanceProfiler initialized")
    
    def start_trace(self, trace_id: str, description: str = ""):
        """Start performance trace."""
        self.active_traces[trace_id] = {
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "description": description
        }
        
        if HAS_PROFILING:
            self.active_traces[trace_id]["memory_snapshot"] = tracemalloc.take_snapshot()
    
    def end_trace(self, trace_id: str) -> Dict[str, Any]:
        """End performance trace and return results."""
        if trace_id not in self.active_traces:
            return {}
        
        trace_start = self.active_traces.pop(trace_id)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - trace_start["start_time"]
        memory_delta = end_memory - trace_start["start_memory"]
        
        profile = {
            "trace_id": trace_id,
            "description": trace_start["description"],
            "duration": duration,
            "memory_start": trace_start["start_memory"],
            "memory_end": end_memory,
            "memory_delta": memory_delta,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Memory profiling
        if HAS_PROFILING and "memory_snapshot" in trace_start:
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(
                trace_start["memory_snapshot"], 'lineno'
            )
            
            profile["memory_top_allocations"] = [
                {
                    "file": stat.traceback.format()[0],
                    "size": stat.size,
                    "size_diff": stat.size_diff
                }
                for stat in top_stats[:10]
            ]
        
        self.profiles[trace_id] = profile
        return profile
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        if not HAS_PROFILING:
            return {"error": "psutil not available"}
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
            "disk_percent": (disk.used / disk.total) * 100,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance profiles."""
        return self.profiles
    
    def clear_profiles(self):
        """Clear all performance profiles."""
        self.profiles.clear()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if HAS_PROFILING:
            return psutil.Process().memory_info().rss
        return 0


def profile_performance(description: str = ""):
    """
    Decorator for profiling function performance.
    
    Args:
        description: Description of the profiled operation.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = getattr(wrapper, '_profiler', None)
            if not profiler:
                profiler = PerformanceProfiler()
                wrapper._profiler = profiler
            
            trace_id = f"{func.__name__}_{int(time.time() * 1000)}"
            profiler.start_trace(trace_id, description or func.__name__)
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
            finally:
                profiler.end_trace(trace_id)
        
        return wrapper
    return decorator


class CacheManager:
    """
    Comprehensive cache management system.
    
    Provides multi-level caching, performance monitoring, and optimization.
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches: Dict[str, Cache] = {}
        self.profiler = PerformanceProfiler()
        self.default_cache: Optional[Cache] = None
        
        # Setup default caches
        self._setup_default_caches()
        
        logger.info("CacheManager initialized")
    
    def _setup_default_caches(self):
        """Setup default cache hierarchy."""
        # L1: Memory cache
        memory_cache = MemoryCache(
            max_size=1000,
            max_memory=50 * 1024 * 1024,  # 50MB
            strategy=CacheStrategy.LRU,
            default_ttl=300  # 5 minutes
        )
        
        self.caches["memory"] = memory_cache
        self.default_cache = memory_cache
        
        # L2: Redis cache (if available)
        if HAS_REDIS:
            try:
                redis_cache = RedisCache(
                    default_ttl=3600,  # 1 hour
                    serializer="pickle"
                )
                
                self.caches["redis"] = redis_cache
                
                # Create multi-level cache
                multi_cache = MultiLevelCache([
                    (CacheLevel.L1_MEMORY, memory_cache),
                    (CacheLevel.L2_REDIS, redis_cache)
                ])
                
                self.caches["multi"] = multi_cache
                self.default_cache = multi_cache
                
            except Exception as e:
                logger.warning(f"Failed to setup Redis cache: {e}")
    
    def add_cache(self, name: str, cache: Cache):
        """Add cache instance."""
        self.caches[name] = cache
        
        logger.info(f"Cache '{name}' added")
    
    def get_cache(self, name: str = None) -> Optional[Cache]:
        """Get cache instance."""
        if name is None:
            return self.default_cache
        
        return self.caches.get(name)
    
    async def connect_all(self):
        """Connect all caches that require connection."""
        for name, cache in self.caches.items():
            if hasattr(cache, 'connect'):
                try:
                    await cache.connect()
                    logger.info(f"Cache '{name}' connected")
                except Exception as e:
                    logger.error(f"Failed to connect cache '{name}': {e}")
    
    async def disconnect_all(self):
        """Disconnect all caches."""
        for name, cache in self.caches.items():
            if hasattr(cache, 'disconnect'):
                try:
                    await cache.disconnect()
                    logger.info(f"Cache '{name}' disconnected")
                except Exception as e:
                    logger.error(f"Failed to disconnect cache '{name}': {e}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        
        for name, cache in self.caches.items():
            try:
                stats[name] = cache.get_stats()
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        # Add system metrics
        stats["system"] = self.profiler.get_system_metrics()
        
        return stats
    
    async def warm_up_cache(self, cache_name: str, data_loader: Callable):
        """Warm up cache with initial data."""
        cache = self.get_cache(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        trace_id = f"warmup_{cache_name}"
        self.profiler.start_trace(trace_id, f"Cache warmup for {cache_name}")
        
        try:
            data = await data_loader() if asyncio.iscoroutinefunction(data_loader) else data_loader()
            
            if isinstance(data, dict):
                for key, value in data.items():
                    await cache.set(key, value)
            
            logger.info(f"Cache '{cache_name}' warmed up with {len(data) if isinstance(data, dict) else 1} entries")
            
        finally:
            self.profiler.end_trace(trace_id)
    
    def optimize_memory(self):
        """Optimize memory usage."""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory stats
        if HAS_PROFILING:
            memory_info = psutil.Process().memory_info()
            
            logger.info(f"Memory optimization: collected {collected} objects, RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        return {
            "collected_objects": collected,
            "memory_info": psutil.Process().memory_info()._asdict() if HAS_PROFILING else {}
        }


def create_cache_manager() -> CacheManager:
    """
    Create configured cache manager.
    
    Returns:
        Configured cache manager.
    """
    return CacheManager()