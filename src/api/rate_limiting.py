"""
API Rate Limiting and Throttling System.

This module provides:
- Token bucket rate limiting
- User-based rate limiting
- API key management
- Request throttling and queuing
- Rate limit monitoring and metrics
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max burst requests
    
    def __post_init__(self):
        """Validate rate limit configuration."""
        if self.requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")
        if self.requests_per_hour < 0:
            raise ValueError("requests_per_hour must be non-negative")
        if self.requests_per_day < 0:
            raise ValueError("requests_per_day must be non-negative")
        if self.burst_limit < 0:
            raise ValueError("burst_limit must be non-negative")


@dataclass
class APIKey:
    """API key configuration."""
    key: str
    name: str
    user_id: str
    rate_limit: RateLimit
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.usage_count += 1


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens.
            refill_rate: Tokens per second refill rate.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume.
            
        Returns:
            True if tokens were consumed, False if insufficient tokens.
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def available_tokens(self) -> int:
        """Get number of available tokens."""
        with self.lock:
            self._refill()
            return int(self.tokens)


class RateLimiter:
    """Comprehensive rate limiting system."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.api_keys: Dict[str, APIKey] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.lock = threading.Lock()
        
        logger.info("RateLimiter initialized")
    
    def add_api_key(self, api_key: APIKey):
        """
        Add API key to rate limiter.
        
        Args:
            api_key: API key configuration.
        """
        with self.lock:
            self.api_keys[api_key.key] = api_key
            
            # Initialize token buckets for this API key
            self.buckets[api_key.key] = {
                'minute': TokenBucket(
                    capacity=api_key.rate_limit.burst_limit,
                    refill_rate=api_key.rate_limit.requests_per_minute / 60.0
                ),
                'hour': TokenBucket(
                    capacity=api_key.rate_limit.requests_per_hour,
                    refill_rate=api_key.rate_limit.requests_per_hour / 3600.0
                ),
                'day': TokenBucket(
                    capacity=api_key.rate_limit.requests_per_day,
                    refill_rate=api_key.rate_limit.requests_per_day / 86400.0
                )
            }
        
        logger.info(f"API key added: {api_key.name} for user {api_key.user_id}")
    
    def remove_api_key(self, api_key: str):
        """
        Remove API key from rate limiter.
        
        Args:
            api_key: API key to remove.
        """
        with self.lock:
            if api_key in self.api_keys:
                del self.api_keys[api_key]
                del self.buckets[api_key]
                if api_key in self.request_history:
                    del self.request_history[api_key]
        
        logger.info(f"API key removed: {api_key}")
    
    def check_rate_limit(self, api_key: str, endpoint: str = "default") -> Dict[str, Any]:
        """
        Check rate limit for API key.
        
        Args:
            api_key: API key to check.
            endpoint: API endpoint (for endpoint-specific limits).
            
        Returns:
            Rate limit status information.
        """
        if api_key not in self.api_keys:
            return {
                "allowed": False,
                "reason": "Invalid API key",
                "retry_after": None
            }
        
        key_config = self.api_keys[api_key]
        
        if not key_config.enabled:
            return {
                "allowed": False,
                "reason": "API key disabled",
                "retry_after": None
            }
        
        # Check token buckets
        buckets = self.buckets[api_key]
        
        # Check minute limit
        if not buckets['minute'].consume():
            return {
                "allowed": False,
                "reason": "Rate limit exceeded (per minute)",
                "retry_after": 60,
                "limit_type": "minute",
                "available_tokens": buckets['minute'].available_tokens()
            }
        
        # Check hour limit
        if not buckets['hour'].consume():
            # Restore minute token since we didn't complete the request
            buckets['minute'].tokens += 1
            return {
                "allowed": False,
                "reason": "Rate limit exceeded (per hour)",
                "retry_after": 3600,
                "limit_type": "hour",
                "available_tokens": buckets['hour'].available_tokens()
            }
        
        # Check day limit
        if not buckets['day'].consume():
            # Restore tokens since we didn't complete the request
            buckets['minute'].tokens += 1
            buckets['hour'].tokens += 1
            return {
                "allowed": False,
                "reason": "Rate limit exceeded (per day)",
                "retry_after": 86400,
                "limit_type": "day",
                "available_tokens": buckets['day'].available_tokens()
            }
        
        # Update API key usage
        key_config.update_usage()
        
        # Record request
        self.request_history[api_key].append({
            "timestamp": datetime.now(),
            "endpoint": endpoint
        })
        
        return {
            "allowed": True,
            "remaining": {
                "minute": buckets['minute'].available_tokens(),
                "hour": buckets['hour'].available_tokens(),
                "day": buckets['day'].available_tokens()
            }
        }
    
    def get_usage_stats(self, api_key: str) -> Dict[str, Any]:
        """
        Get usage statistics for API key.
        
        Args:
            api_key: API key to get stats for.
            
        Returns:
            Usage statistics.
        """
        if api_key not in self.api_keys:
            return {"error": "API key not found"}
        
        key_config = self.api_keys[api_key]
        buckets = self.buckets[api_key]
        history = self.request_history[api_key]
        
        # Calculate recent request rates
        now = datetime.now()
        recent_requests = [
            req for req in history 
            if now - req["timestamp"] < timedelta(hours=1)
        ]
        
        return {
            "api_key_name": key_config.name,
            "user_id": key_config.user_id,
            "total_usage": key_config.usage_count,
            "last_used": key_config.last_used.isoformat() if key_config.last_used else None,
            "enabled": key_config.enabled,
            "recent_requests": len(recent_requests),
            "remaining_tokens": {
                "minute": buckets['minute'].available_tokens(),
                "hour": buckets['hour'].available_tokens(),
                "day": buckets['day'].available_tokens()
            },
            "rate_limits": {
                "requests_per_minute": key_config.rate_limit.requests_per_minute,
                "requests_per_hour": key_config.rate_limit.requests_per_hour,
                "requests_per_day": key_config.rate_limit.requests_per_day,
                "burst_limit": key_config.rate_limit.burst_limit
            }
        }
    
    def get_all_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for all API keys.
        
        Returns:
            Usage statistics for all keys.
        """
        stats = {}
        
        for api_key in self.api_keys:
            stats[api_key] = self.get_usage_stats(api_key)
        
        return {
            "total_keys": len(self.api_keys),
            "active_keys": sum(1 for key in self.api_keys.values() if key.enabled),
            "api_keys": stats
        }


class RequestThrottler:
    """Request throttling and queuing system."""
    
    def __init__(self, max_concurrent: int = 100):
        """
        Initialize request throttler.
        
        Args:
            max_concurrent: Maximum concurrent requests.
        """
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.request_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = {
            "total_requests": 0,
            "queued_requests": 0,
            "rejected_requests": 0,
            "average_wait_time": 0.0
        }
        
        logger.info(f"RequestThrottler initialized with max_concurrent: {max_concurrent}")
    
    async def throttle_request(self, request_handler: Callable, *args, **kwargs) -> Any:
        """
        Throttle request execution.
        
        Args:
            request_handler: Function to execute.
            *args: Positional arguments for handler.
            **kwargs: Keyword arguments for handler.
            
        Returns:
            Result from request handler.
        """
        start_time = time.time()
        
        async with self.semaphore:
            self.active_requests += 1
            self.metrics["total_requests"] += 1
            
            try:
                # Execute the request
                if asyncio.iscoroutinefunction(request_handler):
                    result = await request_handler(*args, **kwargs)
                else:
                    result = request_handler(*args, **kwargs)
                
                # Update metrics
                wait_time = time.time() - start_time
                self._update_average_wait_time(wait_time)
                
                return result
                
            finally:
                self.active_requests -= 1
    
    def _update_average_wait_time(self, wait_time: float):
        """Update average wait time metric."""
        current_avg = self.metrics["average_wait_time"]
        total_requests = self.metrics["total_requests"]
        
        # Calculate new average
        self.metrics["average_wait_time"] = (
            (current_avg * (total_requests - 1) + wait_time) / total_requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get throttling metrics.
        
        Returns:
            Throttling metrics.
        """
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "queue_size": self.request_queue.qsize(),
            **self.metrics
        }


class APIRateLimitMiddleware:
    """Middleware for API rate limiting."""
    
    def __init__(self, rate_limiter: RateLimiter, throttler: RequestThrottler):
        """
        Initialize middleware.
        
        Args:
            rate_limiter: Rate limiter instance.
            throttler: Request throttler instance.
        """
        self.rate_limiter = rate_limiter
        self.throttler = throttler
        
        logger.info("APIRateLimitMiddleware initialized")
    
    async def __call__(self, request, call_next):
        """
        Process request through rate limiting middleware.
        
        Args:
            request: HTTP request.
            call_next: Next middleware in chain.
            
        Returns:
            HTTP response.
        """
        # Extract API key from request
        api_key = self._extract_api_key(request)
        
        if not api_key:
            return self._create_error_response(
                status_code=401,
                message="API key required",
                error_code="MISSING_API_KEY"
            )
        
        # Check rate limit
        rate_limit_result = self.rate_limiter.check_rate_limit(
            api_key, 
            endpoint=request.url.path
        )
        
        if not rate_limit_result["allowed"]:
            return self._create_rate_limit_response(rate_limit_result)
        
        # Throttle request
        try:
            response = await self.throttler.throttle_request(call_next, request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_limit_result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return self._create_error_response(
                status_code=500,
                message="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def _extract_api_key(self, request) -> Optional[str]:
        """Extract API key from request."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check query parameter
        return request.query_params.get("api_key")
    
    def _create_error_response(self, status_code: int, message: str, error_code: str):
        """Create error response."""
        # This would return appropriate response for the framework being used
        return {
            "status_code": status_code,
            "content": {
                "error_code": error_code,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_rate_limit_response(self, rate_limit_result: Dict[str, Any]):
        """Create rate limit exceeded response."""
        return {
            "status_code": 429,
            "content": {
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": rate_limit_result["reason"],
                "retry_after": rate_limit_result.get("retry_after"),
                "limit_type": rate_limit_result.get("limit_type"),
                "timestamp": datetime.now().isoformat()
            },
            "headers": {
                "Retry-After": str(rate_limit_result.get("retry_after", 60)),
                "X-RateLimit-Remaining": str(rate_limit_result.get("available_tokens", 0))
            }
        }
    
    def _add_rate_limit_headers(self, response, rate_limit_result: Dict[str, Any]):
        """Add rate limit headers to response."""
        if "remaining" in rate_limit_result:
            remaining = rate_limit_result["remaining"]
            response.headers["X-RateLimit-Remaining-Minute"] = str(remaining.get("minute", 0))
            response.headers["X-RateLimit-Remaining-Hour"] = str(remaining.get("hour", 0))
            response.headers["X-RateLimit-Remaining-Day"] = str(remaining.get("day", 0))


def create_rate_limiter() -> RateLimiter:
    """
    Create rate limiter instance.
    
    Returns:
        Rate limiter instance.
    """
    return RateLimiter()


def create_request_throttler(max_concurrent: int = 100) -> RequestThrottler:
    """
    Create request throttler instance.
    
    Args:
        max_concurrent: Maximum concurrent requests.
        
    Returns:
        Request throttler instance.
    """
    return RequestThrottler(max_concurrent)


def create_default_rate_limits() -> Dict[str, RateLimit]:
    """
    Create default rate limit configurations.
    
    Returns:
        Dictionary of rate limit configurations.
    """
    return {
        "free": RateLimit(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_limit=5
        ),
        "basic": RateLimit(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=10
        ),
        "premium": RateLimit(
            requests_per_minute=300,
            requests_per_hour=10000,
            requests_per_day=100000,
            burst_limit=50
        ),
        "enterprise": RateLimit(
            requests_per_minute=1000,
            requests_per_hour=50000,
            requests_per_day=1000000,
            burst_limit=200
        )
    }