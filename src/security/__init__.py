"""
Comprehensive security framework for Nuclear Fusion Analyzer.

This module provides authentication, authorization, JWT tokens, OAuth2 integration,
API key management, rate limiting, and security middleware for the ML platform.
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

# JWT and cryptography imports with fallback
try:
    import jwt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import bcrypt
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# OAuth2 imports with fallback
try:
    from authlib.integrations.starlette_client import OAuth
    from authlib.integrations.starlette_client import OAuthError
    HAS_OAUTH = True
except ImportError:
    HAS_OAUTH = False

# Rate limiting imports with fallback
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

import re
from functools import wraps

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    SCIENTIST = "scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class PermissionLevel(str, Enum):
    """Permission level enumeration."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    RESET = "reset"


@dataclass
class SecurityConfig:
    """Security configuration container."""
    
    # JWT settings
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Password settings
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_hash_rounds: int = 12
    
    # API key settings
    api_key_length: int = 32
    api_key_prefix: str = "nfa_"
    api_key_expire_days: int = 365
    
    # Rate limiting settings
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 100
    rate_limit_window_minutes: int = 15
    
    # OAuth2 settings
    oauth2_providers: Dict[str, Dict[str, str]] = None
    
    # Session settings
    session_expire_hours: int = 24
    session_secret_key: str = secrets.token_urlsafe(32)
    
    # Security headers
    enable_security_headers: bool = True
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.oauth2_providers is None:
            self.oauth2_providers = {}
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8080"]


@dataclass
class User:
    """User model for authentication and authorization."""
    
    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole
    permissions: List[str]
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class APIKey:
    """API key model for API authentication."""
    
    id: str
    name: str
    key_hash: str
    user_id: str
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class PasswordValidator:
    """
    Password validation utility.
    
    Validates password strength according to security policies.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize password validator.
        
        Args:
            config: Security configuration.
        """
        self.config = config
        
    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Check length
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        # Check uppercase
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Check lowercase
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Check numbers
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        # Check special characters
        if self.config.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check common passwords
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a stronger password")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list."""
        common_passwords = {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "dragon", "password1"
        }
        
        return password.lower() in common_passwords


class PasswordManager:
    """
    Password hashing and verification manager.
    
    Uses bcrypt for secure password hashing.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize password manager.
        
        Args:
            config: Security configuration.
        """
        self.config = config
        self.validator = PasswordValidator(config)
        
        if not HAS_CRYPTO:
            logger.warning("Cryptography packages not available, password security reduced")
    
    def hash_password(self, password: str) -> str:
        """
        Hash password with bcrypt.
        
        Args:
            password: Plain text password.
            
        Returns:
            Hashed password.
        """
        if not HAS_CRYPTO:
            # Fallback to simple hashing (insecure)
            return hashlib.sha256(password.encode()).hexdigest()
        
        # Validate password first
        is_valid, errors = self.validator.validate(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {', '.join(errors)}")
        
        # Hash with bcrypt
        salt = bcrypt.gensalt(rounds=self.config.password_hash_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password.
            hashed_password: Hashed password.
            
        Returns:
            True if password matches.
        """
        if not HAS_CRYPTO:
            # Fallback verification
            return hashlib.sha256(password.encode()).hexdigest() == hashed_password
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class JWTManager:
    """
    JWT token management for authentication.
    
    Handles token creation, validation, and refresh.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize JWT manager.
        
        Args:
            config: Security configuration.
        """
        self.config = config
        
        if not HAS_CRYPTO:
            logger.warning("JWT packages not available, token security reduced")
    
    def create_access_token(self, user: User) -> str:
        """
        Create access token for user.
        
        Args:
            user: User object.
            
        Returns:
            JWT access token.
        """
        if not HAS_CRYPTO:
            # Fallback token (insecure)
            return f"fallback_token_{user.id}_{int(time.time())}"
        
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self.config.jwt_access_token_expire_minutes
        )
        
        payload = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'permissions': user.permissions,
            'token_type': TokenType.ACCESS.value,
            'exp': expires_at,
            'iat': datetime.now(timezone.utc),
            'iss': 'fusion_analyzer'
        }
        
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """
        Create refresh token for user.
        
        Args:
            user: User object.
            
        Returns:
            JWT refresh token.
        """
        if not HAS_CRYPTO:
            return f"fallback_refresh_{user.id}_{int(time.time())}"
        
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.config.jwt_refresh_token_expire_days
        )
        
        payload = {
            'user_id': user.id,
            'token_type': TokenType.REFRESH.value,
            'exp': expires_at,
            'iat': datetime.now(timezone.utc),
            'iss': 'fusion_analyzer'
        }
        
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token.
            
        Returns:
            Decoded payload or None if invalid.
        """
        if not HAS_CRYPTO:
            # Fallback verification
            if token.startswith("fallback_token_"):
                parts = token.split("_")
                if len(parts) >= 3:
                    return {'user_id': parts[2], 'token_type': 'access'}
            return None
        
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                issuer='fusion_analyzer'
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token.
            
        Returns:
            New access token or None.
        """
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get('token_type') != TokenType.REFRESH.value:
            return None
        
        # Would need to fetch user from database here
        # For now, create a minimal token
        user_id = payload['user_id']
        
        new_payload = {
            'user_id': user_id,
            'token_type': TokenType.ACCESS.value,
            'exp': datetime.now(timezone.utc) + timedelta(
                minutes=self.config.jwt_access_token_expire_minutes
            ),
            'iat': datetime.now(timezone.utc),
            'iss': 'fusion_analyzer'
        }
        
        if HAS_CRYPTO:
            return jwt.encode(
                new_payload,
                self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm
            )
        else:
            return f"fallback_token_{user_id}_{int(time.time())}"


class APIKeyManager:
    """
    API key management for API authentication.
    
    Handles API key generation, validation, and management.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize API key manager.
        
        Args:
            config: Security configuration.
        """
        self.config = config
        self.api_keys: Dict[str, APIKey] = {}  # In-memory storage
    
    def generate_api_key(self, 
                        user_id: str,
                        name: str,
                        permissions: List[str],
                        expires_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """
        Generate new API key.
        
        Args:
            user_id: User ID.
            name: API key name.
            permissions: List of permissions.
            expires_days: Expiration in days.
            
        Returns:
            Tuple of (raw_key, api_key_object).
        """
        # Generate random key
        raw_key = secrets.token_urlsafe(self.config.api_key_length)
        full_key = f"{self.config.api_key_prefix}{raw_key}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        elif self.config.api_key_expire_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=self.config.api_key_expire_days)
        
        # Create API key object
        api_key = APIKey(
            id=str(uuid.uuid4()),
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        # Store API key
        self.api_keys[key_hash] = api_key
        
        logger.info(f"Generated API key: {name} for user {user_id}")
        return full_key, api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate API key.
        
        Args:
            api_key: Raw API key.
            
        Returns:
            APIKey object if valid, None otherwise.
        """
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up API key
        stored_key = self.api_keys.get(key_hash)
        
        if not stored_key:
            return None
        
        # Check if active
        if not stored_key.is_active:
            return None
        
        # Check expiration
        if stored_key.expires_at and stored_key.expires_at < datetime.now(timezone.utc):
            return None
        
        # Update usage
        stored_key.last_used = datetime.now(timezone.utc)
        stored_key.usage_count += 1
        
        return stored_key
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke API key.
        
        Args:
            key_id: API key ID.
            
        Returns:
            Success status.
        """
        for api_key in self.api_keys.values():
            if api_key.id == key_id:
                api_key.is_active = False
                logger.info(f"Revoked API key: {api_key.name}")
                return True
        
        return False
    
    def list_api_keys(self, user_id: str) -> List[APIKey]:
        """
        List API keys for user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of API keys.
        """
        return [
            api_key for api_key in self.api_keys.values()
            if api_key.user_id == user_id
        ]


class RateLimiter:
    """
    Rate limiting system for API protection.
    
    Implements sliding window rate limiting with Redis backend.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Security configuration.
        """
        self.config = config
        self.redis_client = None
        self.memory_store: Dict[str, List[float]] = {}  # Fallback storage
        
    async def initialize(self, redis_url: Optional[str] = None):
        """
        Initialize Redis connection.
        
        Args:
            redis_url: Redis connection URL.
        """
        if HAS_REDIS and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                await self.redis_client.ping()
                logger.info("Rate limiter initialized with Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        else:
            logger.info("Rate limiter using memory storage")
    
    async def is_allowed(self, 
                        identifier: str,
                        requests_per_minute: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP, user_id, API key).
            requests_per_minute: Custom rate limit.
            
        Returns:
            Tuple of (is_allowed, rate_limit_info).
        """
        limit = requests_per_minute or self.config.rate_limit_requests_per_minute
        window_seconds = 60  # 1 minute window
        
        now = time.time()
        window_start = now - window_seconds
        
        if self.redis_client:
            return await self._check_redis_rate_limit(identifier, limit, window_start, now)
        else:
            return self._check_memory_rate_limit(identifier, limit, window_start, now)
    
    async def _check_redis_rate_limit(self, 
                                     identifier: str,
                                     limit: int,
                                     window_start: float,
                                     now: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis."""
        key = f"rate_limit:{identifier}"
        
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, 60)
        
        results = await pipe.execute()
        current_count = results[1]
        
        is_allowed = current_count < limit
        
        rate_limit_info = {
            'limit': limit,
            'remaining': max(0, limit - current_count - 1),
            'reset_time': int(now) + 60,
            'current_count': current_count
        }
        
        return is_allowed, rate_limit_info
    
    def _check_memory_rate_limit(self,
                                identifier: str,
                                limit: int,
                                window_start: float,
                                now: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using memory storage."""
        if identifier not in self.memory_store:
            self.memory_store[identifier] = []
        
        requests = self.memory_store[identifier]
        
        # Remove old requests
        self.memory_store[identifier] = [
            req_time for req_time in requests if req_time > window_start
        ]
        
        current_count = len(self.memory_store[identifier])
        is_allowed = current_count < limit
        
        if is_allowed:
            self.memory_store[identifier].append(now)
        
        rate_limit_info = {
            'limit': limit,
            'remaining': max(0, limit - current_count - (1 if is_allowed else 0)),
            'reset_time': int(now) + 60,
            'current_count': current_count
        }
        
        return is_allowed, rate_limit_info


class AuthenticationMiddleware:
    """
    Authentication middleware for request processing.
    
    Handles JWT token validation, API key authentication,
    and user context injection.
    """
    
    def __init__(self, 
                 security_config: SecurityConfig,
                 jwt_manager: JWTManager,
                 api_key_manager: APIKeyManager):
        """
        Initialize authentication middleware.
        
        Args:
            security_config: Security configuration.
            jwt_manager: JWT manager instance.
            api_key_manager: API key manager instance.
        """
        self.config = security_config
        self.jwt_manager = jwt_manager
        self.api_key_manager = api_key_manager
        
    async def authenticate_request(self, request) -> Optional[Dict[str, Any]]:
        """
        Authenticate incoming request.
        
        Args:
            request: HTTP request object.
            
        Returns:
            User context or None.
        """
        # Check for JWT token in Authorization header
        auth_header = getattr(request, 'headers', {}).get('Authorization', '')
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            payload = self.jwt_manager.verify_token(token)
            
            if payload:
                return {
                    'user_id': payload.get('user_id'),
                    'username': payload.get('username'),
                    'role': payload.get('role'),
                    'permissions': payload.get('permissions', []),
                    'auth_type': 'jwt'
                }
        
        # Check for API key in header
        api_key_header = getattr(request, 'headers', {}).get('X-API-Key', '')
        
        if api_key_header:
            api_key = self.api_key_manager.validate_api_key(api_key_header)
            
            if api_key:
                return {
                    'user_id': api_key.user_id,
                    'api_key_id': api_key.id,
                    'permissions': api_key.permissions,
                    'auth_type': 'api_key'
                }
        
        return None


class AuthorizationManager:
    """
    Authorization manager for permission checking.
    
    Handles role-based access control and resource permissions.
    """
    
    def __init__(self):
        """Initialize authorization manager."""
        # Define role hierarchy
        self.role_hierarchy = {
            UserRole.ADMIN: [UserRole.SCIENTIST, UserRole.ANALYST, UserRole.VIEWER, UserRole.API_USER],
            UserRole.SCIENTIST: [UserRole.ANALYST, UserRole.VIEWER],
            UserRole.ANALYST: [UserRole.VIEWER],
            UserRole.VIEWER: [],
            UserRole.API_USER: []
        }
        
        # Define default permissions per role
        self.role_permissions = {
            UserRole.ADMIN: ['*'],  # All permissions
            UserRole.SCIENTIST: [
                'experiments:read', 'experiments:write', 'experiments:delete',
                'models:read', 'models:write', 'data:read', 'data:write',
                'analytics:read', 'analytics:write'
            ],
            UserRole.ANALYST: [
                'experiments:read', 'models:read', 'data:read',
                'analytics:read', 'analytics:write'
            ],
            UserRole.VIEWER: [
                'experiments:read', 'models:read', 'data:read', 'analytics:read'
            ],
            UserRole.API_USER: [
                'api:read', 'models:read', 'predictions:write'
            ]
        }
    
    def check_permission(self, 
                        user_context: Dict[str, Any],
                        required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_context: User authentication context.
            required_permission: Required permission string.
            
        Returns:
            True if authorized.
        """
        user_role = user_context.get('role')
        user_permissions = user_context.get('permissions', [])
        
        # Check for admin wildcard
        if '*' in user_permissions:
            return True
        
        # Check explicit permission
        if required_permission in user_permissions:
            return True
        
        # Check role-based permissions
        if user_role and user_role in self.role_permissions:
            role_perms = self.role_permissions[UserRole(user_role)]
            if '*' in role_perms or required_permission in role_perms:
                return True
        
        return False
    
    def get_user_permissions(self, user_role: UserRole) -> List[str]:
        """
        Get all permissions for a user role.
        
        Args:
            user_role: User role.
            
        Returns:
            List of permissions.
        """
        permissions = set()
        
        # Add direct permissions
        if user_role in self.role_permissions:
            permissions.update(self.role_permissions[user_role])
        
        # Add inherited permissions
        if user_role in self.role_hierarchy:
            for inherited_role in self.role_hierarchy[user_role]:
                if inherited_role in self.role_permissions:
                    permissions.update(self.role_permissions[inherited_role])
        
        return list(permissions)


def create_security_system(config: SecurityConfig) -> Dict[str, Any]:
    """
    Create complete security system.
    
    Args:
        config: Security configuration.
        
    Returns:
        Dictionary with security components.
    """
    password_manager = PasswordManager(config)
    jwt_manager = JWTManager(config)
    api_key_manager = APIKeyManager(config)
    rate_limiter = RateLimiter(config)
    auth_middleware = AuthenticationMiddleware(config, jwt_manager, api_key_manager)
    auth_manager = AuthorizationManager()
    
    return {
        'config': config,
        'password_manager': password_manager,
        'jwt_manager': jwt_manager,
        'api_key_manager': api_key_manager,
        'rate_limiter': rate_limiter,
        'auth_middleware': auth_middleware,
        'auth_manager': auth_manager
    }