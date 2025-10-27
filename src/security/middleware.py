"""
Security middleware and decorators for request processing.

This module provides middleware for authentication, authorization,
rate limiting, security headers, and CORS protection.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from functools import wraps
import re

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """
    Security headers configuration and management.
    
    Implements security headers to protect against common web vulnerabilities.
    """
    
    def __init__(self, 
                 strict_transport_security: bool = True,
                 content_type_options: bool = True,
                 frame_options: bool = True,
                 xss_protection: bool = True,
                 referrer_policy: bool = True,
                 content_security_policy: Optional[str] = None):
        """
        Initialize security headers.
        
        Args:
            strict_transport_security: Enable HSTS header.
            content_type_options: Enable X-Content-Type-Options header.
            frame_options: Enable X-Frame-Options header.
            xss_protection: Enable X-XSS-Protection header.
            referrer_policy: Enable Referrer-Policy header.
            content_security_policy: CSP header value.
        """
        self.headers = {}
        
        if strict_transport_security:
            self.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        if content_type_options:
            self.headers['X-Content-Type-Options'] = 'nosniff'
        
        if frame_options:
            self.headers['X-Frame-Options'] = 'DENY'
        
        if xss_protection:
            self.headers['X-XSS-Protection'] = '1; mode=block'
        
        if referrer_policy:
            self.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        if content_security_policy:
            self.headers['Content-Security-Policy'] = content_security_policy
        
        # Additional security headers
        self.headers['X-Permitted-Cross-Domain-Policies'] = 'none'
        self.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        self.headers['Pragma'] = 'no-cache'
        self.headers['Expires'] = '0'
    
    def get_headers(self) -> Dict[str, str]:
        """Get security headers dictionary."""
        return self.headers.copy()


class CORSConfig:
    """
    CORS (Cross-Origin Resource Sharing) configuration.
    
    Manages CORS policies for secure cross-origin requests.
    """
    
    def __init__(self,
                 allowed_origins: List[str] = None,
                 allowed_methods: List[str] = None,
                 allowed_headers: List[str] = None,
                 exposed_headers: List[str] = None,
                 allow_credentials: bool = False,
                 max_age: int = 3600):
        """
        Initialize CORS configuration.
        
        Args:
            allowed_origins: List of allowed origins.
            allowed_methods: List of allowed HTTP methods.
            allowed_headers: List of allowed headers.
            exposed_headers: List of headers to expose.
            allow_credentials: Whether to allow credentials.
            max_age: Preflight cache duration.
        """
        self.allowed_origins = allowed_origins or ["http://localhost:3000"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.exposed_headers = exposed_headers or []
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed.
        
        Args:
            origin: Request origin.
            
        Returns:
            True if allowed.
        """
        if "*" in self.allowed_origins:
            return True
        
        return origin in self.allowed_origins
    
    def get_cors_headers(self, origin: str, method: str) -> Dict[str, str]:
        """
        Get CORS headers for response.
        
        Args:
            origin: Request origin.
            method: Request method.
            
        Returns:
            CORS headers dictionary.
        """
        headers = {}
        
        if self.is_origin_allowed(origin):
            headers['Access-Control-Allow-Origin'] = origin
        
        if method == 'OPTIONS':
            headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            headers['Access-Control-Max-Age'] = str(self.max_age)
        
        if self.exposed_headers:
            headers['Access-Control-Expose-Headers'] = ', '.join(self.exposed_headers)
        
        if self.allow_credentials:
            headers['Access-Control-Allow-Credentials'] = 'true'
        
        return headers


class RequestValidator:
    """
    Request validation and sanitization.
    
    Validates and sanitizes incoming requests for security.
    """
    
    def __init__(self):
        """Initialize request validator."""
        # Common attack patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\bOR\s+\d+\s*=\s*\d+)",
            r"(\'\s*(OR|AND)\s*\'\d+\'\s*=\s*\'\d+)",
        ]
        
        self.xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<object[^>]*>.*?</object>)",
            r"(<embed[^>]*>.*?</embed>)",
        ]
        
        self.path_traversal_patterns = [
            r"(\.\./)",
            r"(\.\.\\)",
            r"(%2e%2e%2f)",
            r"(%2e%2e\\)",
        ]
    
    def validate_request_path(self, path: str) -> bool:
        """
        Validate request path for security.
        
        Args:
            path: Request path.
            
        Returns:
            True if path is safe.
        """
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {path}")
                return False
        
        # Check for null bytes
        if '\x00' in path:
            logger.warning(f"Null byte in path: {path}")
            return False
        
        return True
    
    def validate_query_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate query parameters.
        
        Args:
            params: Query parameters.
            
        Returns:
            True if parameters are safe.
        """
        for key, value in params.items():
            if isinstance(value, str):
                # Check for SQL injection
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        logger.warning(f"SQL injection attempt in param {key}: {value}")
                        return False
                
                # Check for XSS
                for pattern in self.xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        logger.warning(f"XSS attempt in param {key}: {value}")
                        return False
        
        return True
    
    def sanitize_input(self, data: str) -> str:
        """
        Sanitize input data.
        
        Args:
            data: Input data to sanitize.
            
        Returns:
            Sanitized data.
        """
        # Remove null bytes
        data = data.replace('\x00', '')
        
        # Basic HTML entity encoding
        data = data.replace('<', '&lt;')
        data = data.replace('>', '&gt;')
        data = data.replace('"', '&quot;')
        data = data.replace("'", '&#x27;')
        data = data.replace('/', '&#x2F;')
        
        return data


class SecurityMiddleware:
    """
    Main security middleware class.
    
    Combines all security components into a unified middleware.
    """
    
    def __init__(self,
                 auth_manager,
                 rate_limiter,
                 security_headers: Optional[SecurityHeaders] = None,
                 cors_config: Optional[CORSConfig] = None,
                 request_validator: Optional[RequestValidator] = None):
        """
        Initialize security middleware.
        
        Args:
            auth_manager: Authentication manager.
            rate_limiter: Rate limiter instance.
            security_headers: Security headers configuration.
            cors_config: CORS configuration.
            request_validator: Request validator.
        """
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.security_headers = security_headers or SecurityHeaders()
        self.cors_config = cors_config or CORSConfig()
        self.request_validator = request_validator or RequestValidator()
        
        logger.info("SecurityMiddleware initialized")
    
    async def process_request(self, request) -> Optional[Dict[str, Any]]:
        """
        Process incoming request through security pipeline.
        
        Args:
            request: HTTP request object.
            
        Returns:
            Security context or None if request should be rejected.
        """
        # Extract request information
        method = getattr(request, 'method', 'GET')
        path = getattr(request, 'url', {}).get('path', '/')
        headers = getattr(request, 'headers', {})
        query_params = getattr(request, 'query_params', {})
        
        # Get client identifier for rate limiting
        client_ip = self._get_client_ip(request)
        user_agent = headers.get('user-agent', '')
        
        # 1. Validate request path
        if not self.request_validator.validate_request_path(path):
            return None
        
        # 2. Validate query parameters
        if not self.request_validator.validate_query_params(dict(query_params)):
            return None
        
        # 3. Check rate limiting
        rate_limit_key = f"{client_ip}:{user_agent[:50]}"
        is_allowed, rate_info = await self.rate_limiter.is_allowed(rate_limit_key)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return {
                'error': 'rate_limit_exceeded',
                'rate_limit_info': rate_info
            }
        
        # 4. Authenticate request
        auth_context = await self.auth_manager.authenticate_request(request)
        
        # 5. Create security context
        security_context = {
            'client_ip': client_ip,
            'user_agent': user_agent,
            'rate_limit_info': rate_info,
            'auth_context': auth_context,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return security_context
    
    def process_response(self, request, response) -> Dict[str, str]:
        """
        Process outgoing response with security headers.
        
        Args:
            request: HTTP request object.
            response: HTTP response object.
            
        Returns:
            Security headers to add.
        """
        headers = {}
        
        # Add security headers
        headers.update(self.security_headers.get_headers())
        
        # Add CORS headers
        origin = getattr(request, 'headers', {}).get('origin', '')
        method = getattr(request, 'method', 'GET')
        
        if origin:
            cors_headers = self.cors_config.get_cors_headers(origin, method)
            headers.update(cors_headers)
        
        return headers
    
    def _get_client_ip(self, request) -> str:
        """Extract client IP from request."""
        headers = getattr(request, 'headers', {})
        
        # Check forwarded headers
        forwarded_for = headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to client host
        client = getattr(request, 'client', {})
        return client.get('host', '127.0.0.1')


def require_auth(permissions: List[str] = None):
    """
    Decorator to require authentication for endpoint.
    
    Args:
        permissions: Required permissions list.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if hasattr(arg, 'headers'):
                    request = arg
                    break
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Check for security context
            security_context = getattr(request, 'state', {}).get('security_context')
            
            if not security_context:
                return {'error': 'Security context not found', 'status_code': 500}
            
            auth_context = security_context.get('auth_context')
            
            if not auth_context:
                return {'error': 'Authentication required', 'status_code': 401}
            
            # Check permissions if specified
            if permissions:
                user_permissions = auth_context.get('permissions', [])
                
                for required_permission in permissions:
                    if required_permission not in user_permissions and '*' not in user_permissions:
                        return {'error': 'Insufficient permissions', 'status_code': 403}
            
            # Add auth context to kwargs
            kwargs['auth_context'] = auth_context
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(roles: List[str]):
    """
    Decorator to require specific user role.
    
    Args:
        roles: Required roles list.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if hasattr(arg, 'headers'):
                    request = arg
                    break
            
            if not request:
                raise ValueError("Request object not found")
            
            security_context = getattr(request, 'state', {}).get('security_context')
            auth_context = security_context.get('auth_context') if security_context else None
            
            if not auth_context:
                return {'error': 'Authentication required', 'status_code': 401}
            
            user_role = auth_context.get('role')
            
            if user_role not in roles:
                return {'error': 'Insufficient role permissions', 'status_code': 403}
            
            kwargs['auth_context'] = auth_context
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int):
    """
    Decorator to apply rate limiting to endpoint.
    
    Args:
        requests_per_minute: Rate limit threshold.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if hasattr(arg, 'headers'):
                    request = arg
                    break
            
            if not request:
                return await func(*args, **kwargs)
            
            security_context = getattr(request, 'state', {}).get('security_context')
            
            if security_context:
                rate_info = security_context.get('rate_limit_info', {})
                
                if rate_info.get('current_count', 0) >= requests_per_minute:
                    return {
                        'error': 'Rate limit exceeded for this endpoint',
                        'status_code': 429,
                        'rate_limit_info': rate_info
                    }
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_input(fields: List[str] = None):
    """
    Decorator to sanitize input data.
    
    Args:
        fields: Fields to sanitize (all if None).
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            validator = RequestValidator()
            
            # Sanitize kwargs
            for key, value in kwargs.items():
                if fields is None or key in fields:
                    if isinstance(value, str):
                        kwargs[key] = validator.sanitize_input(value)
                    elif isinstance(value, dict):
                        kwargs[key] = {
                            k: validator.sanitize_input(v) if isinstance(v, str) else v
                            for k, v in value.items()
                        }
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityContextManager:
    """
    Security context manager for request lifecycle.
    
    Manages security context throughout request processing.
    """
    
    def __init__(self):
        """Initialize security context manager."""
        self.contexts: Dict[str, Dict[str, Any]] = {}
    
    def create_context(self, request_id: str, context_data: Dict[str, Any]):
        """
        Create security context for request.
        
        Args:
            request_id: Unique request identifier.
            context_data: Security context data.
        """
        self.contexts[request_id] = {
            **context_data,
            'created_at': datetime.now(timezone.utc),
            'request_id': request_id
        }
    
    def get_context(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get security context.
        
        Args:
            request_id: Request identifier.
            
        Returns:
            Security context or None.
        """
        return self.contexts.get(request_id)
    
    def update_context(self, request_id: str, updates: Dict[str, Any]):
        """
        Update security context.
        
        Args:
            request_id: Request identifier.
            updates: Context updates.
        """
        if request_id in self.contexts:
            self.contexts[request_id].update(updates)
            self.contexts[request_id]['updated_at'] = datetime.now(timezone.utc)
    
    def cleanup_context(self, request_id: str):
        """
        Clean up security context.
        
        Args:
            request_id: Request identifier.
        """
        self.contexts.pop(request_id, None)
    
    def cleanup_expired_contexts(self, max_age_minutes: int = 60):
        """
        Clean up expired contexts.
        
        Args:
            max_age_minutes: Maximum context age in minutes.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        
        expired_ids = [
            req_id for req_id, context in self.contexts.items()
            if context.get('created_at', datetime.min.replace(tzinfo=timezone.utc)) < cutoff
        ]
        
        for req_id in expired_ids:
            self.cleanup_context(req_id)