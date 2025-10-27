"""
Encryption and cryptographic utilities for secure data handling.

This module provides encryption, hashing, digital signatures,
and other cryptographic operations for data protection.
"""

import os
import base64
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from dataclasses import dataclass
import json

# Cryptography imports with fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

logger = logging.getLogger(__name__)


@dataclass
class EncryptionConfig:
    """Encryption configuration container."""
    
    # Symmetric encryption
    symmetric_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    
    # Asymmetric encryption
    rsa_key_size: int = 4096
    signing_algorithm: str = "Ed25519"
    
    # Key rotation
    key_rotation_days: int = 90
    max_key_age_days: int = 365
    
    # Storage
    key_storage_path: str = "keys"
    encrypted_data_path: str = "encrypted_data"


class SymmetricEncryption:
    """
    Symmetric encryption utilities using Fernet (AES-128-CBC).
    
    Provides secure symmetric encryption for data at rest and in transit.
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize symmetric encryption.
        
        Args:
            key: Encryption key. Generated if None.
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        if key is None:
            key = Fernet.generate_key()
        
        self.key = key
        self.fernet = Fernet(key)
        
        logger.info("SymmetricEncryption initialized")
    
    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate new encryption key.
        
        Returns:
            Encryption key bytes.
        """
        return Fernet.generate_key()
    
    @classmethod
    def derive_key_from_password(cls, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password string.
            salt: Salt bytes. Generated if None.
            
        Returns:
            Tuple of (key, salt).
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        if salt is None:
            salt = os.urandom(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        return key, salt
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt.
            
        Returns:
            Encrypted data bytes.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data bytes.
            
        Returns:
            Decrypted data bytes.
        """
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_json(self, obj: Any) -> bytes:
        """
        Encrypt JSON serializable object.
        
        Args:
            obj: Object to encrypt.
            
        Returns:
            Encrypted JSON bytes.
        """
        json_data = json.dumps(obj, default=str)
        return self.encrypt(json_data)
    
    def decrypt_json(self, encrypted_data: bytes) -> Any:
        """
        Decrypt JSON data.
        
        Args:
            encrypted_data: Encrypted JSON bytes.
            
        Returns:
            Decrypted object.
        """
        decrypted_data = self.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode('utf-8'))


class AsymmetricEncryption:
    """
    Asymmetric encryption utilities using RSA.
    
    Provides public/private key encryption for secure communication.
    """
    
    def __init__(self, private_key: Optional[rsa.RSAPrivateKey] = None, public_key: Optional[rsa.RSAPublicKey] = None):
        """
        Initialize asymmetric encryption.
        
        Args:
            private_key: RSA private key.
            public_key: RSA public key.
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        self.private_key = private_key
        self.public_key = public_key
        
        logger.info("AsymmetricEncryption initialized")
    
    @classmethod
    def generate_key_pair(cls, key_size: int = 4096) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generate RSA key pair.
        
        Args:
            key_size: Key size in bits.
            
        Returns:
            Tuple of (private_key, public_key).
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def encrypt_with_public_key(self, data: Union[str, bytes], public_key: Optional[rsa.RSAPublicKey] = None) -> bytes:
        """
        Encrypt data with public key.
        
        Args:
            data: Data to encrypt.
            public_key: Public key to use. Uses instance key if None.
            
        Returns:
            Encrypted data bytes.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = public_key or self.public_key
        if not key:
            raise ValueError("No public key available")
        
        encrypted = key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def decrypt_with_private_key(self, encrypted_data: bytes, private_key: Optional[rsa.RSAPrivateKey] = None) -> bytes:
        """
        Decrypt data with private key.
        
        Args:
            encrypted_data: Encrypted data bytes.
            private_key: Private key to use. Uses instance key if None.
            
        Returns:
            Decrypted data bytes.
        """
        key = private_key or self.private_key
        if not key:
            raise ValueError("No private key available")
        
        decrypted = key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted
    
    def serialize_private_key(self, password: Optional[str] = None) -> bytes:
        """
        Serialize private key to PEM format.
        
        Args:
            password: Optional password for key encryption.
            
        Returns:
            Serialized private key bytes.
        """
        if not self.private_key:
            raise ValueError("No private key available")
        
        encryption_algorithm = serialization.NoEncryption()
        if password:
            encryption_algorithm = serialization.BestAvailableEncryption(password.encode())
        
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
    
    def serialize_public_key(self) -> bytes:
        """
        Serialize public key to PEM format.
        
        Returns:
            Serialized public key bytes.
        """
        if not self.public_key:
            raise ValueError("No public key available")
        
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    @classmethod
    def load_private_key(cls, key_data: bytes, password: Optional[str] = None) -> rsa.RSAPrivateKey:
        """
        Load private key from PEM data.
        
        Args:
            key_data: PEM key data.
            password: Optional password.
            
        Returns:
            Private key object.
        """
        password_bytes = password.encode() if password else None
        
        return serialization.load_pem_private_key(
            key_data,
            password=password_bytes,
            backend=default_backend()
        )
    
    @classmethod
    def load_public_key(cls, key_data: bytes) -> rsa.RSAPublicKey:
        """
        Load public key from PEM data.
        
        Args:
            key_data: PEM key data.
            
        Returns:
            Public key object.
        """
        return serialization.load_pem_public_key(
            key_data,
            backend=default_backend()
        )


class DigitalSignature:
    """
    Digital signature utilities for data integrity and authentication.
    
    Uses Ed25519 for fast and secure digital signatures.
    """
    
    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey] = None):
        """
        Initialize digital signature.
        
        Args:
            private_key: Ed25519 private key.
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        self.private_key = private_key
        self.public_key = private_key.public_key() if private_key else None
        
        logger.info("DigitalSignature initialized")
    
    @classmethod
    def generate_key_pair(cls) -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
        """
        Generate Ed25519 key pair.
        
        Returns:
            Tuple of (private_key, public_key).
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("Cryptography library not available")
        
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def sign(self, data: Union[str, bytes]) -> bytes:
        """
        Sign data with private key.
        
        Args:
            data: Data to sign.
            
        Returns:
            Signature bytes.
        """
        if not self.private_key:
            raise ValueError("No private key available for signing")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.private_key.sign(data)
    
    def verify(self, data: Union[str, bytes], signature: bytes, public_key: Optional[ed25519.Ed25519PublicKey] = None) -> bool:
        """
        Verify signature with public key.
        
        Args:
            data: Original data.
            signature: Signature to verify.
            public_key: Public key to use. Uses instance key if None.
            
        Returns:
            True if signature is valid.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = public_key or self.public_key
        if not key:
            raise ValueError("No public key available for verification")
        
        try:
            key.verify(signature, data)
            return True
        except InvalidSignature:
            return False


class HashingUtilities:
    """
    Cryptographic hashing utilities.
    
    Provides secure hashing functions for data integrity.
    """
    
    @staticmethod
    def sha256(data: Union[str, bytes]) -> str:
        """
        Compute SHA-256 hash.
        
        Args:
            data: Data to hash.
            
        Returns:
            Hex-encoded hash string.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def sha512(data: Union[str, bytes]) -> str:
        """
        Compute SHA-512 hash.
        
        Args:
            data: Data to hash.
            
        Returns:
            Hex-encoded hash string.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha512(data).hexdigest()
    
    @staticmethod
    def blake2b(data: Union[str, bytes], key: Optional[bytes] = None) -> str:
        """
        Compute BLAKE2b hash.
        
        Args:
            data: Data to hash.
            key: Optional key for keyed hashing.
            
        Returns:
            Hex-encoded hash string.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.blake2b(data, key=key).hexdigest()
    
    @staticmethod
    def hmac_sha256(data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """
        Compute HMAC-SHA256.
        
        Args:
            data: Data to authenticate.
            key: Secret key.
            
        Returns:
            Hex-encoded HMAC string.
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    @staticmethod
    def verify_hmac(data: Union[str, bytes], key: Union[str, bytes], expected_hmac: str) -> bool:
        """
        Verify HMAC.
        
        Args:
            data: Original data.
            key: Secret key.
            expected_hmac: Expected HMAC value.
            
        Returns:
            True if HMAC is valid.
        """
        computed_hmac = HashingUtilities.hmac_sha256(data, key)
        return hmac.compare_digest(computed_hmac, expected_hmac)


class SecureRandom:
    """
    Cryptographically secure random number generation.
    
    Provides secure random values for cryptographic operations.
    """
    
    @staticmethod
    def bytes(length: int) -> bytes:
        """
        Generate random bytes.
        
        Args:
            length: Number of bytes to generate.
            
        Returns:
            Random bytes.
        """
        return secrets.token_bytes(length)
    
    @staticmethod
    def hex(length: int) -> str:
        """
        Generate random hex string.
        
        Args:
            length: Number of bytes (hex string will be 2x length).
            
        Returns:
            Random hex string.
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def urlsafe(length: int) -> str:
        """
        Generate URL-safe random string.
        
        Args:
            length: Number of bytes (string will be longer due to encoding).
            
        Returns:
            URL-safe random string.
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def integer(min_value: int, max_value: int) -> int:
        """
        Generate random integer in range.
        
        Args:
            min_value: Minimum value (inclusive).
            max_value: Maximum value (inclusive).
            
        Returns:
            Random integer.
        """
        return secrets.randbelow(max_value - min_value + 1) + min_value


class KeyManager:
    """
    Cryptographic key management system.
    
    Handles key generation, storage, rotation, and lifecycle management.
    """
    
    def __init__(self, config: EncryptionConfig):
        """
        Initialize key manager.
        
        Args:
            config: Encryption configuration.
        """
        self.config = config
        self.keys: Dict[str, Dict[str, Any]] = {}
        
        # Create storage directories
        os.makedirs(config.key_storage_path, exist_ok=True)
        
        logger.info("KeyManager initialized")
    
    def generate_symmetric_key(self, key_id: str, description: str = "") -> bytes:
        """
        Generate and store symmetric key.
        
        Args:
            key_id: Unique key identifier.
            description: Key description.
            
        Returns:
            Generated key bytes.
        """
        key = SymmetricEncryption.generate_key()
        
        key_info = {
            'key_id': key_id,
            'type': 'symmetric',
            'algorithm': 'Fernet',
            'created_at': datetime.now(timezone.utc),
            'description': description,
            'key_data': base64.b64encode(key).decode('ascii')
        }
        
        self.keys[key_id] = key_info
        self._save_key(key_id, key_info)
        
        logger.info(f"Generated symmetric key: {key_id}")
        return key
    
    def generate_asymmetric_key_pair(self, key_id: str, description: str = "") -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generate and store asymmetric key pair.
        
        Args:
            key_id: Unique key identifier.
            description: Key description.
            
        Returns:
            Tuple of (private_key, public_key).
        """
        private_key, public_key = AsymmetricEncryption.generate_key_pair(self.config.rsa_key_size)
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        key_info = {
            'key_id': key_id,
            'type': 'asymmetric',
            'algorithm': 'RSA',
            'key_size': self.config.rsa_key_size,
            'created_at': datetime.now(timezone.utc),
            'description': description,
            'private_key': base64.b64encode(private_pem).decode('ascii'),
            'public_key': base64.b64encode(public_pem).decode('ascii')
        }
        
        self.keys[key_id] = key_info
        self._save_key(key_id, key_info)
        
        logger.info(f"Generated asymmetric key pair: {key_id}")
        return private_key, public_key
    
    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Get key information.
        
        Args:
            key_id: Key identifier.
            
        Returns:
            Key information or None.
        """
        if key_id in self.keys:
            return self.keys[key_id]
        
        # Try to load from storage
        return self._load_key(key_id)
    
    def rotate_key(self, key_id: str) -> bool:
        """
        Rotate encryption key.
        
        Args:
            key_id: Key identifier.
            
        Returns:
            Success status.
        """
        old_key = self.get_key(key_id)
        if not old_key:
            return False
        
        # Archive old key
        archive_id = f"{key_id}_archived_{int(datetime.now().timestamp())}"
        old_key['archived_at'] = datetime.now(timezone.utc)
        self.keys[archive_id] = old_key
        self._save_key(archive_id, old_key)
        
        # Generate new key
        if old_key['type'] == 'symmetric':
            self.generate_symmetric_key(key_id, old_key['description'])
        elif old_key['type'] == 'asymmetric':
            self.generate_asymmetric_key_pair(key_id, old_key['description'])
        
        logger.info(f"Rotated key: {key_id}")
        return True
    
    def _save_key(self, key_id: str, key_info: Dict[str, Any]):
        """Save key to storage."""
        key_path = os.path.join(self.config.key_storage_path, f"{key_id}.json")
        
        # Encrypt key data before storage
        symmetric_key = SymmetricEncryption.generate_key()
        encryptor = SymmetricEncryption(symmetric_key)
        
        key_info_copy = key_info.copy()
        key_info_copy['created_at'] = key_info_copy['created_at'].isoformat()
        
        encrypted_data = encryptor.encrypt_json(key_info_copy)
        
        with open(key_path, 'wb') as f:
            f.write(encrypted_data)
    
    def _load_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Load key from storage."""
        key_path = os.path.join(self.config.key_storage_path, f"{key_id}.json")
        
        if not os.path.exists(key_path):
            return None
        
        try:
            with open(key_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Would need to decrypt with master key in practice
            # For now, assume unencrypted storage
            
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to load key {key_id}: {e}")
            return None


def create_crypto_system(config: EncryptionConfig) -> Dict[str, Any]:
    """
    Create complete cryptographic system.
    
    Args:
        config: Encryption configuration.
        
    Returns:
        Dictionary with crypto components.
    """
    key_manager = KeyManager(config)
    
    # Generate default system keys
    symmetric_key = key_manager.generate_symmetric_key(
        'system_default',
        'Default system encryption key'
    )
    
    private_key, public_key = key_manager.generate_asymmetric_key_pair(
        'system_rsa',
        'Default system RSA key pair'
    )
    
    signing_private, signing_public = DigitalSignature.generate_key_pair()
    
    return {
        'key_manager': key_manager,
        'symmetric': SymmetricEncryption(symmetric_key),
        'asymmetric': AsymmetricEncryption(private_key, public_key),
        'signature': DigitalSignature(signing_private),
        'hashing': HashingUtilities(),
        'random': SecureRandom()
    }