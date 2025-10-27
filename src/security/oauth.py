"""
OAuth2 integration and external authentication providers.

This module provides OAuth2 integration with popular providers
like Google, GitHub, Microsoft, and others for seamless authentication.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import secrets
import urllib.parse

# OAuth2 and HTTP client imports with fallback
try:
    import httpx
    from authlib.integrations.starlette_client import OAuth, OAuthError
    from authlib.common.security import generate_token
    HAS_OAUTH = True
except ImportError:
    HAS_OAUTH = False

logger = logging.getLogger(__name__)


class OAuthProvider(str, Enum):
    """OAuth provider enumeration."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    ORCID = "orcid"
    LINKEDIN = "linkedin"
    CUSTOM = "custom"


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    
    provider: OAuthProvider
    client_id: str
    client_secret: str
    server_metadata_url: Optional[str] = None
    authorize_url: Optional[str] = None
    access_token_url: Optional[str] = None
    userinfo_url: Optional[str] = None
    scopes: List[str] = None
    redirect_uri: str = ""
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []


@dataclass
class OAuthUserInfo:
    """OAuth user information."""
    
    provider: OAuthProvider
    provider_user_id: str
    email: str
    name: str
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    profile_url: Optional[str] = None
    verified_email: bool = False
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


class OAuthProviderManager:
    """
    OAuth provider management and configuration.
    
    Handles multiple OAuth providers with standardized interfaces.
    """
    
    def __init__(self):
        """Initialize OAuth provider manager."""
        self.providers: Dict[OAuthProvider, OAuthConfig] = {}
        self.oauth_client = None
        
        if HAS_OAUTH:
            self.oauth_client = OAuth()
        else:
            logger.warning("OAuth dependencies not available")
        
        logger.info("OAuthProviderManager initialized")
    
    def register_provider(self, config: OAuthConfig):
        """
        Register OAuth provider.
        
        Args:
            config: OAuth provider configuration.
        """
        self.providers[config.provider] = config
        
        if self.oauth_client and HAS_OAUTH:
            self._configure_authlib_provider(config)
        
        logger.info(f"Registered OAuth provider: {config.provider}")
    
    def _configure_authlib_provider(self, config: OAuthConfig):
        """Configure provider with Authlib."""
        provider_config = {
            'client_id': config.client_id,
            'client_secret': config.client_secret,
        }
        
        # Add provider-specific configurations
        if config.provider == OAuthProvider.GOOGLE:
            provider_config.update({
                'server_metadata_url': config.server_metadata_url or 'https://accounts.google.com/.well-known/openid_configuration',
                'client_kwargs': {
                    'scope': ' '.join(config.scopes or ['openid', 'email', 'profile'])
                }
            })
        
        elif config.provider == OAuthProvider.GITHUB:
            provider_config.update({
                'access_token_url': config.access_token_url or 'https://github.com/login/oauth/access_token',
                'authorize_url': config.authorize_url or 'https://github.com/login/oauth/authorize',
                'api_base_url': 'https://api.github.com/',
                'client_kwargs': {
                    'scope': ' '.join(config.scopes or ['user:email'])
                }
            })
        
        elif config.provider == OAuthProvider.MICROSOFT:
            provider_config.update({
                'server_metadata_url': config.server_metadata_url or 'https://login.microsoftonline.com/common/v2.0/.well-known/openid_configuration',
                'client_kwargs': {
                    'scope': ' '.join(config.scopes or ['openid', 'email', 'profile'])
                }
            })
        
        elif config.provider == OAuthProvider.ORCID:
            provider_config.update({
                'access_token_url': config.access_token_url or 'https://orcid.org/oauth/token',
                'authorize_url': config.authorize_url or 'https://orcid.org/oauth/authorize',
                'api_base_url': 'https://pub.orcid.org/v3.0/',
                'client_kwargs': {
                    'scope': ' '.join(config.scopes or ['/authenticate'])
                }
            })
        
        # Register with OAuth client
        self.oauth_client.register(
            name=config.provider.value,
            **provider_config
        )
    
    def get_authorization_url(self, 
                             provider: OAuthProvider,
                             redirect_uri: str,
                             state: Optional[str] = None) -> Tuple[str, str]:
        """
        Get authorization URL for OAuth flow.
        
        Args:
            provider: OAuth provider.
            redirect_uri: Redirect URI after authorization.
            state: Optional state parameter.
            
        Returns:
            Tuple of (authorization_url, state).
        """
        if not self.oauth_client or not HAS_OAUTH:
            raise RuntimeError("OAuth not available")
        
        if provider not in self.providers:
            raise ValueError(f"Provider not registered: {provider}")
        
        if state is None:
            state = generate_token()
        
        client = getattr(self.oauth_client, provider.value)
        
        authorization_url = client.authorize_redirect_url(
            redirect_uri=redirect_uri,
            state=state
        )
        
        return authorization_url, state
    
    async def exchange_code_for_token(self, 
                                     provider: OAuthProvider,
                                     code: str,
                                     redirect_uri: str,
                                     state: Optional[str] = None) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.
        
        Args:
            provider: OAuth provider.
            code: Authorization code.
            redirect_uri: Redirect URI.
            state: State parameter.
            
        Returns:
            Token response.
        """
        if not self.oauth_client or not HAS_OAUTH:
            raise RuntimeError("OAuth not available")
        
        if provider not in self.providers:
            raise ValueError(f"Provider not registered: {provider}")
        
        client = getattr(self.oauth_client, provider.value)
        
        token = await client.authorize_access_token(
            code=code,
            redirect_uri=redirect_uri,
            state=state
        )
        
        return token
    
    async def get_user_info(self, 
                           provider: OAuthProvider,
                           access_token: str) -> OAuthUserInfo:
        """
        Get user information from OAuth provider.
        
        Args:
            provider: OAuth provider.
            access_token: Access token.
            
        Returns:
            User information.
        """
        if provider == OAuthProvider.GOOGLE:
            return await self._get_google_user_info(access_token)
        elif provider == OAuthProvider.GITHUB:
            return await self._get_github_user_info(access_token)
        elif provider == OAuthProvider.MICROSOFT:
            return await self._get_microsoft_user_info(access_token)
        elif provider == OAuthProvider.ORCID:
            return await self._get_orcid_user_info(access_token)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _get_google_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from Google."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://www.googleapis.com/userinfo/v2/me',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            response.raise_for_status()
            
            data = response.json()
            
            return OAuthUserInfo(
                provider=OAuthProvider.GOOGLE,
                provider_user_id=data['id'],
                email=data['email'],
                name=data['name'],
                username=data.get('email'),
                avatar_url=data.get('picture'),
                profile_url=data.get('link'),
                verified_email=data.get('verified_email', False),
                raw_data=data
            )
    
    async def _get_github_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from GitHub."""
        async with httpx.AsyncClient() as client:
            # Get user data
            user_response = await client.get(
                'https://api.github.com/user',
                headers={'Authorization': f'token {access_token}'}
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            
            # Get email data
            email_response = await client.get(
                'https://api.github.com/user/emails',
                headers={'Authorization': f'token {access_token}'}
            )
            email_response.raise_for_status()
            emails = email_response.json()
            
            # Find primary/verified email
            primary_email = None
            for email in emails:
                if email.get('primary') and email.get('verified'):
                    primary_email = email['email']
                    break
            
            if not primary_email and emails:
                primary_email = emails[0]['email']
            
            return OAuthUserInfo(
                provider=OAuthProvider.GITHUB,
                provider_user_id=str(user_data['id']),
                email=primary_email or '',
                name=user_data.get('name') or user_data['login'],
                username=user_data['login'],
                avatar_url=user_data.get('avatar_url'),
                profile_url=user_data.get('html_url'),
                verified_email=bool(primary_email),
                raw_data={'user': user_data, 'emails': emails}
            )
    
    async def _get_microsoft_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from Microsoft."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://graph.microsoft.com/v1.0/me',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            response.raise_for_status()
            
            data = response.json()
            
            return OAuthUserInfo(
                provider=OAuthProvider.MICROSOFT,
                provider_user_id=data['id'],
                email=data.get('mail') or data.get('userPrincipalName', ''),
                name=data.get('displayName', ''),
                username=data.get('userPrincipalName'),
                verified_email=True,  # Microsoft emails are typically verified
                raw_data=data
            )
    
    async def _get_orcid_user_info(self, access_token: str) -> OAuthUserInfo:
        """Get user info from ORCID."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://pub.orcid.org/v3.0/person',
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'Accept': 'application/json'
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract name
            name = ""
            if 'name' in data and data['name']:
                given_names = data['name'].get('given-names', {}).get('value', '')
                family_name = data['name'].get('family-name', {}).get('value', '')
                name = f"{given_names} {family_name}".strip()
            
            # Extract email
            email = ""
            if 'emails' in data and data['emails'].get('email'):
                emails = data['emails']['email']
                for email_obj in emails:
                    if email_obj.get('primary') or not email:
                        email = email_obj.get('email', '')
            
            # Extract ORCID ID
            orcid_id = data.get('orcid-identifier', {}).get('path', '')
            
            return OAuthUserInfo(
                provider=OAuthProvider.ORCID,
                provider_user_id=orcid_id,
                email=email,
                name=name,
                username=orcid_id,
                profile_url=f"https://orcid.org/{orcid_id}" if orcid_id else None,
                verified_email=bool(email),
                raw_data=data
            )


class OAuthStateManager:
    """
    OAuth state management for security.
    
    Manages state tokens to prevent CSRF attacks.
    """
    
    def __init__(self, expire_minutes: int = 10):
        """
        Initialize state manager.
        
        Args:
            expire_minutes: State expiration time in minutes.
        """
        self.expire_minutes = expire_minutes
        self.states: Dict[str, Dict[str, Any]] = {}
        
        logger.info("OAuthStateManager initialized")
    
    def create_state(self, 
                    provider: OAuthProvider,
                    redirect_uri: str,
                    user_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create OAuth state token.
        
        Args:
            provider: OAuth provider.
            redirect_uri: Redirect URI.
            user_data: Optional user data to store.
            
        Returns:
            State token.
        """
        state = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.expire_minutes)
        
        self.states[state] = {
            'provider': provider,
            'redirect_uri': redirect_uri,
            'user_data': user_data or {},
            'created_at': datetime.now(timezone.utc),
            'expires_at': expires_at
        }
        
        # Clean up expired states
        self._cleanup_expired_states()
        
        return state
    
    def validate_state(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Validate OAuth state token.
        
        Args:
            state: State token.
            
        Returns:
            State data if valid, None otherwise.
        """
        if state not in self.states:
            return None
        
        state_data = self.states[state]
        
        # Check expiration
        if state_data['expires_at'] < datetime.now(timezone.utc):
            del self.states[state]
            return None
        
        # Remove state after use
        del self.states[state]
        
        return state_data
    
    def _cleanup_expired_states(self):
        """Remove expired states."""
        now = datetime.now(timezone.utc)
        expired_states = [
            state for state, data in self.states.items()
            if data['expires_at'] < now
        ]
        
        for state in expired_states:
            del self.states[state]


class SocialAuthenticationManager:
    """
    Social authentication manager.
    
    Integrates OAuth providers with user management.
    """
    
    def __init__(self):
        """Initialize social authentication manager."""
        self.provider_manager = OAuthProviderManager()
        self.state_manager = OAuthStateManager()
        self.user_mappings: Dict[str, str] = {}  # provider_user_id -> local_user_id
        
        logger.info("SocialAuthenticationManager initialized")
    
    def configure_providers(self, provider_configs: List[OAuthConfig]):
        """
        Configure OAuth providers.
        
        Args:
            provider_configs: List of provider configurations.
        """
        for config in provider_configs:
            self.provider_manager.register_provider(config)
        
        logger.info(f"Configured {len(provider_configs)} OAuth providers")
    
    async def initiate_oauth_flow(self, 
                                 provider: OAuthProvider,
                                 redirect_uri: str,
                                 user_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Initiate OAuth authentication flow.
        
        Args:
            provider: OAuth provider.
            redirect_uri: Redirect URI.
            user_data: Optional user data.
            
        Returns:
            Tuple of (authorization_url, state).
        """
        # Create state
        state = self.state_manager.create_state(provider, redirect_uri, user_data)
        
        # Get authorization URL
        auth_url, _ = self.provider_manager.get_authorization_url(
            provider, redirect_uri, state
        )
        
        return auth_url, state
    
    async def complete_oauth_flow(self, 
                                 code: str,
                                 state: str) -> Optional[OAuthUserInfo]:
        """
        Complete OAuth authentication flow.
        
        Args:
            code: Authorization code.
            state: State token.
            
        Returns:
            User information if successful.
        """
        # Validate state
        state_data = self.state_manager.validate_state(state)
        if not state_data:
            logger.warning("Invalid or expired OAuth state")
            return None
        
        provider = state_data['provider']
        redirect_uri = state_data['redirect_uri']
        
        try:
            # Exchange code for token
            token_response = await self.provider_manager.exchange_code_for_token(
                provider, code, redirect_uri, state
            )
            
            access_token = token_response.get('access_token')
            if not access_token:
                logger.error("No access token in OAuth response")
                return None
            
            # Get user info
            user_info = await self.provider_manager.get_user_info(provider, access_token)
            
            # Store provider mapping
            self._store_provider_mapping(user_info)
            
            return user_info
            
        except Exception as e:
            logger.error(f"OAuth flow error: {e}")
            return None
    
    def _store_provider_mapping(self, user_info: OAuthUserInfo):
        """Store provider to local user mapping."""
        provider_key = f"{user_info.provider.value}:{user_info.provider_user_id}"
        
        # In a real implementation, this would:
        # 1. Check if provider mapping exists
        # 2. Create new user account if needed
        # 3. Link provider to existing user if email matches
        # 4. Store mapping in database
        
        # For now, just store in memory
        self.user_mappings[provider_key] = user_info.provider_user_id
        
        logger.info(f"Stored provider mapping: {provider_key}")
    
    def get_local_user_id(self, provider: OAuthProvider, provider_user_id: str) -> Optional[str]:
        """
        Get local user ID from provider user ID.
        
        Args:
            provider: OAuth provider.
            provider_user_id: Provider user ID.
            
        Returns:
            Local user ID if found.
        """
        provider_key = f"{provider.value}:{provider_user_id}"
        return self.user_mappings.get(provider_key)
    
    def list_configured_providers(self) -> List[OAuthProvider]:
        """
        List configured OAuth providers.
        
        Returns:
            List of configured providers.
        """
        return list(self.provider_manager.providers.keys())


def create_default_oauth_configs() -> List[OAuthConfig]:
    """
    Create default OAuth configurations.
    
    Returns:
        List of default OAuth configs.
    """
    return [
        OAuthConfig(
            provider=OAuthProvider.GOOGLE,
            client_id="${GOOGLE_CLIENT_ID}",
            client_secret="${GOOGLE_CLIENT_SECRET}",
            scopes=["openid", "email", "profile"]
        ),
        OAuthConfig(
            provider=OAuthProvider.GITHUB,
            client_id="${GITHUB_CLIENT_ID}",
            client_secret="${GITHUB_CLIENT_SECRET}",
            scopes=["user:email"]
        ),
        OAuthConfig(
            provider=OAuthProvider.MICROSOFT,
            client_id="${MICROSOFT_CLIENT_ID}",
            client_secret="${MICROSOFT_CLIENT_SECRET}",
            scopes=["openid", "email", "profile"]
        ),
        OAuthConfig(
            provider=OAuthProvider.ORCID,
            client_id="${ORCID_CLIENT_ID}",
            client_secret="${ORCID_CLIENT_SECRET}",
            scopes=["/authenticate"]
        )
    ]