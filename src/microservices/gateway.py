"""
API Gateway for microservices routing and management.

This module provides:
- Request routing and path rewriting
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning support
- Metrics collection and logging
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Any, Union, Callable, Pattern
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone, timedelta

# FastAPI for gateway
try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.base import BaseHTTPMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# HTTP clients
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

logger = logging.getLogger(__name__)


class RouteStrategy(Enum):
    """Route matching strategies."""
    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"
    WILDCARD = "wildcard"


@dataclass
class RouteRule:
    """API gateway route rule."""
    
    path_pattern: str
    target_service: str
    target_path: Optional[str] = None
    strategy: RouteStrategy = RouteStrategy.PREFIX
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    require_auth: bool = True
    rate_limit: Optional[str] = None  # e.g., "100/minute"
    timeout: int = 30
    retry_count: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    transform_request: Optional[Callable] = None
    transform_response: Optional[Callable] = None
    
    def __post_init__(self):
        """Compile regex pattern if needed."""
        if self.strategy == RouteStrategy.REGEX:
            self.compiled_pattern: Pattern = re.compile(self.path_pattern)
        else:
            self.compiled_pattern = None
    
    def matches(self, path: str, method: str) -> bool:
        """Check if rule matches request."""
        if method not in self.methods:
            return False
        
        if self.strategy == RouteStrategy.EXACT:
            return path == self.path_pattern
        elif self.strategy == RouteStrategy.PREFIX:
            return path.startswith(self.path_pattern)
        elif self.strategy == RouteStrategy.REGEX:
            return bool(self.compiled_pattern.match(path))
        elif self.strategy == RouteStrategy.WILDCARD:
            import fnmatch
            return fnmatch.fnmatch(path, self.path_pattern)
        
        return False
    
    def get_target_path(self, original_path: str) -> str:
        """Get target path for service."""
        if self.target_path:
            if self.strategy == RouteStrategy.PREFIX:
                # Remove prefix and prepend target path
                remaining = original_path[len(self.path_pattern):]
                return f"{self.target_path}{remaining}"
            else:
                return self.target_path
        
        return original_path


@dataclass
class GatewayConfig:
    """API Gateway configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    default_rate_limit: str = "1000/minute"
    rate_limit_storage_url: str = "memory://"
    
    # Timeouts
    default_timeout: int = 30
    connection_timeout: int = 5
    
    # Retry settings
    default_retry_count: int = 3
    retry_backoff: float = 1.0
    
    # Security
    enable_auth: bool = True
    auth_service: str = "auth-service"
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"


class RequestTransformer(ABC):
    """Abstract request transformer."""
    
    @abstractmethod
    async def transform(self, request: Request) -> Request:
        """Transform request before forwarding."""
        pass


class ResponseTransformer(ABC):
    """Abstract response transformer."""
    
    @abstractmethod
    async def transform(self, response: Response) -> Response:
        """Transform response before returning."""
        pass


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API Gateway."""
    
    def __init__(self, app, auth_service_url: str):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application.
            auth_service_url: Authentication service URL.
        """
        super().__init__(app)
        self.auth_service_url = auth_service_url
        self.http_client = httpx.AsyncClient()
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication."""
        # Skip auth for health checks and auth endpoints
        if request.url.path in ["/health", "/metrics"] or request.url.path.startswith("/auth"):
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing authorization header"}
            )
        
        try:
            # Validate token with auth service
            auth_response = await self.http_client.post(
                f"{self.auth_service_url}/validate",
                headers={"Authorization": auth_header},
                timeout=5.0
            )
            
            if auth_response.status_code != 200:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid token"}
                )
            
            # Add user info to request state
            user_info = auth_response.json()
            request.state.user = user_info
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication service unavailable"}
            )
        
        return await call_next(request)


class MetricsCollector:
    """Metrics collection for API Gateway."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_count: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
        self.error_count: Dict[str, int] = {}
        self.active_connections = 0
        
        logger.info("MetricsCollector initialized")
    
    def record_request(self, path: str, method: str):
        """Record incoming request."""
        key = f"{method}:{path}"
        self.request_count[key] = self.request_count.get(key, 0) + 1
        self.active_connections += 1
    
    def record_response(self, path: str, method: str, status_code: int, response_time: float):
        """Record response metrics."""
        key = f"{method}:{path}"
        
        # Response time
        if key not in self.response_times:
            self.response_times[key] = []
        self.response_times[key].append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times[key]) > 1000:
            self.response_times[key] = self.response_times[key][-1000:]
        
        # Error count
        if status_code >= 400:
            self.error_count[key] = self.error_count.get(key, 0) + 1
        
        self.active_connections = max(0, self.active_connections - 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        metrics = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "active_connections": self.active_connections,
            "response_times": {}
        }
        
        # Calculate response time statistics
        for key, times in self.response_times.items():
            if times:
                metrics["response_times"][key] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return metrics


class APIGateway:
    """
    Comprehensive API Gateway for microservices.
    
    Provides routing, authentication, rate limiting, and monitoring.
    """
    
    def __init__(self, config: GatewayConfig, microservice_framework):
        """
        Initialize API Gateway.
        
        Args:
            config: Gateway configuration.
            microservice_framework: Microservices framework instance.
        """
        if not HAS_FASTAPI:
            raise RuntimeError("FastAPI not available for API Gateway")
        
        self.config = config
        self.microservice_framework = microservice_framework
        self.routes: List[RouteRule] = []
        self.app = FastAPI(title="Fusion Analysis API Gateway", version="1.0.0")
        self.metrics = MetricsCollector()
        
        # HTTP client for service calls
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.default_timeout)
        )
        
        # Rate limiter
        if HAS_SLOWAPI:
            self.limiter = Limiter(key_func=get_remote_address)
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("APIGateway initialized")
    
    def _setup_middleware(self):
        """Setup middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allow_origins,
            allow_credentials=True,
            allow_methods=self.config.allow_methods,
            allow_headers=self.config.allow_headers,
        )
        
        # Authentication
        if self.config.enable_auth:
            auth_service_url = f"http://{self.config.auth_service}:8000"
            self.app.add_middleware(AuthenticationMiddleware, auth_service_url=auth_service_url)
        
        # Metrics middleware
        if self.config.enable_metrics:
            @self.app.middleware("http")
            async def metrics_middleware(request: Request, call_next):
                start_time = time.time()
                
                # Record request
                self.metrics.record_request(request.url.path, request.method)
                
                # Process request
                response = await call_next(request)
                
                # Record response
                response_time = time.time() - start_time
                self.metrics.record_response(
                    request.url.path,
                    request.method,
                    response.status_code,
                    response_time
                )
                
                return response
    
    def _setup_routes(self):
        """Setup default routes."""
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Metrics endpoint."""
            return self.metrics.get_metrics()
        
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def route_request(request: Request, path: str):
            """Route requests to microservices."""
            return await self._route_request(request, path)
    
    def add_route(self, rule: RouteRule):
        """Add routing rule."""
        self.routes.append(rule)
        self.routes.sort(key=lambda r: len(r.path_pattern), reverse=True)  # Longest first
        
        logger.info(f"Route added: {rule.path_pattern} -> {rule.target_service}")
    
    def remove_route(self, path_pattern: str):
        """Remove routing rule."""
        self.routes = [r for r in self.routes if r.path_pattern != path_pattern]
        
        logger.info(f"Route removed: {path_pattern}")
    
    async def _route_request(self, request: Request, path: str) -> Response:
        """Route request to appropriate microservice."""
        # Find matching route
        matching_rule = None
        for rule in self.routes:
            if rule.matches(f"/{path}", request.method):
                matching_rule = rule
                break
        
        if not matching_rule:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Check rate limit
        if matching_rule.rate_limit and HAS_SLOWAPI:
            try:
                await self.limiter.limit(matching_rule.rate_limit)(request)
            except RateLimitExceeded:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Discover target service
        service = await self.microservice_framework.discover_service(matching_rule.target_service)
        if not service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # Prepare target URL
        target_path = matching_rule.get_target_path(f"/{path}")
        target_url = f"{service.url}{target_path}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.update(matching_rule.headers)
        
        # Remove hop-by-hop headers
        hop_by_hop = [
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
        ]
        for header in hop_by_hop:
            headers.pop(header, None)
        
        # Get request body
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
        
        # Forward request with retry logic
        for attempt in range(matching_rule.retry_count + 1):
            try:
                response = await self.http_client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=body,
                    timeout=matching_rule.timeout
                )
                
                # Prepare response headers
                response_headers = dict(response.headers)
                
                # Remove hop-by-hop headers
                for header in hop_by_hop:
                    response_headers.pop(header, None)
                
                # Create FastAPI response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response_headers.get("content-type")
                )
                
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < matching_rule.retry_count:
                    await asyncio.sleep(self.config.retry_backoff * (2 ** attempt))
                    continue
                
                raise HTTPException(status_code=502, detail="Bad gateway")
        
        raise HTTPException(status_code=502, detail="Bad gateway")
    
    async def start(self):
        """Start API Gateway."""
        logger.info(f"Starting API Gateway on {self.config.host}:{self.config.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop API Gateway."""
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("API Gateway stopped")


def create_fusion_gateway_routes() -> List[RouteRule]:
    """Create default routes for fusion analysis services."""
    routes = [
        # Authentication service
        RouteRule(
            path_pattern="/auth",
            target_service="auth-service",
            target_path="/auth",
            require_auth=False
        ),
        
        # Data service
        RouteRule(
            path_pattern="/api/v1/data",
            target_service="data-service",
            target_path="/api/v1"
        ),
        
        # ML service
        RouteRule(
            path_pattern="/api/v1/ml",
            target_service="ml-service",
            target_path="/api/v1"
        ),
        
        # Prediction service
        RouteRule(
            path_pattern="/api/v1/predict",
            target_service="prediction-service",
            target_path="/api/v1"
        ),
        
        # Visualization service
        RouteRule(
            path_pattern="/api/v1/viz",
            target_service="visualization-service",
            target_path="/api/v1"
        ),
        
        # Analytics service
        RouteRule(
            path_pattern="/api/v1/analytics",
            target_service="analytics-service",
            target_path="/api/v1"
        ),
        
        # Monitoring service
        RouteRule(
            path_pattern="/api/v1/monitoring",
            target_service="monitoring-service",
            target_path="/api/v1"
        )
    ]
    
    return routes


def create_api_gateway(
    config: Optional[GatewayConfig] = None,
    microservice_framework=None
) -> APIGateway:
    """
    Create configured API Gateway.
    
    Args:
        config: Gateway configuration.
        microservice_framework: Microservices framework.
        
    Returns:
        Configured API Gateway.
    """
    if config is None:
        config = GatewayConfig()
    
    gateway = APIGateway(config, microservice_framework)
    
    # Add default routes
    for route in create_fusion_gateway_routes():
        gateway.add_route(route)
    
    return gateway