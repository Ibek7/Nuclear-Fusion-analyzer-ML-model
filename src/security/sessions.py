"""
Session management for user authentication and state handling.

This module provides secure session management including
user sessions, token blacklisting, and session storage.
"""

import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, asdict
from uuid import uuid4
import logging
from enum import Enum
import hashlib
import secrets

# Redis for session storage (with fallback)
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class SessionData:
    """Session data container."""
    
    session_id: str
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    device_id: Optional[str] = None
    location: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure datetime objects are timezone aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.last_accessed.tzinfo is None:
            self.last_accessed = self.last_accessed.replace(tzinfo=timezone.utc)
        if self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid and active."""
        return (
            self.status == SessionStatus.ACTIVE and
            not self.is_expired()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        data['status'] = self.status.value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['status'] = SessionStatus(data['status'])
        
        return cls(**data)


@dataclass
class SessionConfig:
    """Session management configuration."""
    
    # Session timeouts
    default_timeout_minutes: int = 60
    max_timeout_minutes: int = 480  # 8 hours
    refresh_threshold_minutes: int = 15
    
    # Security settings
    max_sessions_per_user: int = 5
    require_secure_cookies: bool = True
    enable_ip_validation: bool = True
    enable_user_agent_validation: bool = False
    
    # Storage settings
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "fusion_session"
    cleanup_interval_minutes: int = 30
    
    # Advanced features
    enable_concurrent_session_limit: bool = True
    enable_device_tracking: bool = True
    enable_location_tracking: bool = False
    enable_session_events: bool = True


class SessionStore:
    """
    Session storage backend interface.
    
    Provides pluggable storage for session data.
    """
    
    async def save_session(self, session: SessionData) -> bool:
        """Save session data."""
        raise NotImplementedError
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        raise NotImplementedError
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        raise NotImplementedError
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        raise NotImplementedError
    
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    """
    In-memory session storage implementation.
    
    For development and testing purposes.
    """
    
    def __init__(self):
        """Initialize memory store."""
        self.sessions: Dict[str, SessionData] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
        
        logger.info("MemorySessionStore initialized")
    
    async def save_session(self, session: SessionData) -> bool:
        """Save session data."""
        try:
            self.sessions[session.session_id] = session
            
            # Track user sessions
            if session.user_id not in self.user_sessions:
                self.user_sessions[session.user_id] = set()
            self.user_sessions[session.user_id].add(session.session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            session = self.sessions.pop(session_id, None)
            if session:
                # Remove from user sessions
                user_sessions = self.user_sessions.get(session.user_id, set())
                user_sessions.discard(session_id)
                
                if not user_sessions:
                    self.user_sessions.pop(session.user_id, None)
            
            return session is not None
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        session_ids = self.user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired() or session.status != SessionStatus.ACTIVE:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)


class RedisSessionStore(SessionStore):
    """
    Redis-based session storage implementation.
    
    For production use with persistence and scalability.
    """
    
    def __init__(self, config: SessionConfig):
        """
        Initialize Redis store.
        
        Args:
            config: Session configuration.
        """
        if not HAS_REDIS:
            raise RuntimeError("Redis library not available")
        
        self.config = config
        self.redis_client = None
        
        logger.info("RedisSessionStore initialized")
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client connection."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.config.redis_url)
        
        return self.redis_client
    
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.config.redis_prefix}:session:{session_id}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user sessions."""
        return f"{self.config.redis_prefix}:user:{user_id}:sessions"
    
    async def save_session(self, session: SessionData) -> bool:
        """Save session data."""
        try:
            client = await self._get_redis()
            
            # Serialize session data
            session_data = json.dumps(session.to_dict())
            
            # Calculate TTL in seconds
            ttl = int((session.expires_at - datetime.now(timezone.utc)).total_seconds())
            
            if ttl <= 0:
                logger.warning(f"Session {session.session_id} is already expired")
                return False
            
            # Save session data
            session_key = self._session_key(session.session_id)
            await client.setex(session_key, ttl, session_data)
            
            # Add to user sessions
            user_sessions_key = self._user_sessions_key(session.user_id)
            await client.sadd(user_sessions_key, session.session_id)
            await client.expire(user_sessions_key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        try:
            client = await self._get_redis()
            
            session_key = self._session_key(session_id)
            session_data = await client.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            return SessionData.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            client = await self._get_redis()
            
            # Get session to find user ID
            session = await self.get_session(session_id)
            
            # Delete session
            session_key = self._session_key(session_id)
            result = await client.delete(session_key)
            
            # Remove from user sessions
            if session:
                user_sessions_key = self._user_sessions_key(session.user_id)
                await client.srem(user_sessions_key, session_id)
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        try:
            client = await self._get_redis()
            
            user_sessions_key = self._user_sessions_key(user_id)
            session_ids = await client.smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session = await self.get_session(session_id.decode())
                if session:
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []
    
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        # Redis automatically expires keys with TTL
        # This method can be used for additional cleanup
        return 0


class TokenBlacklist:
    """
    Token blacklist for JWT revocation.
    
    Manages revoked tokens to prevent reuse.
    """
    
    def __init__(self, store: SessionStore):
        """
        Initialize token blacklist.
        
        Args:
            store: Session storage backend.
        """
        self.store = store
        self.blacklist: Set[str] = set()
        
        logger.info("TokenBlacklist initialized")
    
    async def add_token(self, token_jti: str, expires_at: datetime):
        """
        Add token to blacklist.
        
        Args:
            token_jti: JWT ID claim.
            expires_at: Token expiration time.
        """
        self.blacklist.add(token_jti)
        
        # In production, store in Redis with TTL
        if isinstance(self.store, RedisSessionStore):
            try:
                client = await self.store._get_redis()
                key = f"{self.store.config.redis_prefix}:blacklist:{token_jti}"
                ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
                
                if ttl > 0:
                    await client.setex(key, ttl, "1")
                    
            except Exception as e:
                logger.error(f"Failed to blacklist token {token_jti}: {e}")
    
    async def is_blacklisted(self, token_jti: str) -> bool:
        """
        Check if token is blacklisted.
        
        Args:
            token_jti: JWT ID claim.
            
        Returns:
            True if token is blacklisted.
        """
        if token_jti in self.blacklist:
            return True
        
        # Check Redis blacklist
        if isinstance(self.store, RedisSessionStore):
            try:
                client = await self.store._get_redis()
                key = f"{self.store.config.redis_prefix}:blacklist:{token_jti}"
                return await client.exists(key) > 0
                
            except Exception as e:
                logger.error(f"Failed to check blacklist for {token_jti}: {e}")
        
        return False


class SessionManager:
    """
    Comprehensive session management system.
    
    Handles session creation, validation, refresh, and cleanup.
    """
    
    def __init__(self, config: SessionConfig, store: Optional[SessionStore] = None):
        """
        Initialize session manager.
        
        Args:
            config: Session configuration.
            store: Session storage backend.
        """
        self.config = config
        self.store = store or MemorySessionStore()
        self.blacklist = TokenBlacklist(self.store)
        
        # Start cleanup task
        self._cleanup_task = None
        
        logger.info("SessionManager initialized")
    
    async def start(self):
        """Start session manager services."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("SessionManager services started")
    
    async def stop(self):
        """Stop session manager services."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("SessionManager services stopped")
    
    async def create_session(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[str],
        permissions: List[str],
        ip_address: str,
        user_agent: str,
        device_id: Optional[str] = None,
        location: Optional[str] = None,
        timeout_minutes: Optional[int] = None
    ) -> SessionData:
        """
        Create new user session.
        
        Args:
            user_id: User identifier.
            username: Username.
            email: User email.
            roles: User roles.
            permissions: User permissions.
            ip_address: Client IP address.
            user_agent: Client user agent.
            device_id: Optional device identifier.
            location: Optional location.
            timeout_minutes: Session timeout in minutes.
            
        Returns:
            Created session data.
        """
        # Check concurrent session limit
        if self.config.enable_concurrent_session_limit:
            user_sessions = await self.store.get_user_sessions(user_id)
            active_sessions = [s for s in user_sessions if s.is_valid()]
            
            if len(active_sessions) >= self.config.max_sessions_per_user:
                # Remove oldest session
                oldest_session = min(active_sessions, key=lambda s: s.last_accessed)
                await self.revoke_session(oldest_session.session_id)
        
        # Create session
        now = datetime.now(timezone.utc)
        timeout = timeout_minutes or self.config.default_timeout_minutes
        
        # Limit timeout to maximum
        timeout = min(timeout, self.config.max_timeout_minutes)
        
        session = SessionData(
            session_id=self._generate_session_id(),
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=now,
            last_accessed=now,
            expires_at=now + timedelta(minutes=timeout),
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
            location=location,
            status=SessionStatus.ACTIVE
        )
        
        # Save session
        await self.store.save_session(session)
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Session data or None.
        """
        session = await self.store.get_session(session_id)
        
        if not session or not session.is_valid():
            return None
        
        return session
    
    async def validate_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[SessionData]:
        """
        Validate session with security checks.
        
        Args:
            session_id: Session identifier.
            ip_address: Client IP address.
            user_agent: Client user agent.
            
        Returns:
            Valid session data or None.
        """
        session = await self.get_session(session_id)
        
        if not session:
            return None
        
        # IP validation
        if self.config.enable_ip_validation and ip_address:
            if session.ip_address != ip_address:
                logger.warning(f"IP mismatch for session {session_id}: {session.ip_address} != {ip_address}")
                await self.revoke_session(session_id)
                return None
        
        # User agent validation
        if self.config.enable_user_agent_validation and user_agent:
            if session.user_agent != user_agent:
                logger.warning(f"User agent mismatch for session {session_id}")
                await self.revoke_session(session_id)
                return None
        
        # Update last accessed time
        session.last_accessed = datetime.now(timezone.utc)
        await self.store.save_session(session)
        
        return session
    
    async def refresh_session(self, session_id: str, extend_minutes: Optional[int] = None) -> bool:
        """
        Refresh session expiration.
        
        Args:
            session_id: Session identifier.
            extend_minutes: Minutes to extend session.
            
        Returns:
            Success status.
        """
        session = await self.get_session(session_id)
        
        if not session:
            return False
        
        # Check if session needs refresh
        time_to_expire = session.expires_at - datetime.now(timezone.utc)
        refresh_threshold = timedelta(minutes=self.config.refresh_threshold_minutes)
        
        if time_to_expire > refresh_threshold:
            return True  # No refresh needed
        
        # Extend session
        extend_by = extend_minutes or self.config.default_timeout_minutes
        extend_by = min(extend_by, self.config.max_timeout_minutes)
        
        session.expires_at = datetime.now(timezone.utc) + timedelta(minutes=extend_by)
        session.last_accessed = datetime.now(timezone.utc)
        
        await self.store.save_session(session)
        
        logger.info(f"Refreshed session {session_id} for {extend_by} minutes")
        return True
    
    async def revoke_session(self, session_id: str) -> bool:
        """
        Revoke session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Success status.
        """
        session = await self.store.get_session(session_id)
        
        if session:
            session.status = SessionStatus.REVOKED
            await self.store.save_session(session)
        
        result = await self.store.delete_session(session_id)
        
        if result:
            logger.info(f"Revoked session {session_id}")
        
        return result
    
    async def revoke_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            user_id: User identifier.
            except_session: Session to keep active.
            
        Returns:
            Number of revoked sessions.
        """
        user_sessions = await self.store.get_user_sessions(user_id)
        revoked_count = 0
        
        for session in user_sessions:
            if session.session_id != except_session:
                await self.revoke_session(session.session_id)
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            List of active sessions.
        """
        all_sessions = await self.store.get_user_sessions(user_id)
        return [s for s in all_sessions if s.is_valid()]
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return secrets.token_urlsafe(32)
    
    async def _cleanup_loop(self):
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                await self.store.cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")


def create_session_manager(config: SessionConfig, use_redis: bool = False) -> SessionManager:
    """
    Create session manager with appropriate storage backend.
    
    Args:
        config: Session configuration.
        use_redis: Whether to use Redis storage.
        
    Returns:
        Configured session manager.
    """
    if use_redis and HAS_REDIS:
        store = RedisSessionStore(config)
    else:
        store = MemorySessionStore()
        if use_redis:
            logger.warning("Redis not available, using memory storage")
    
    return SessionManager(config, store)