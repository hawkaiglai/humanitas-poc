"""
Cryptographic template generation for the HUMANITAS POC system.

This module implements secure biometric template generation using Argon2
password hashing. The templates are designed to be cryptographically secure,
irreversible, and suitable for zero-knowledge proof systems.

The Argon2 algorithm provides resistance against both CPU and GPU-based
attacks while maintaining deterministic output for the same input, making
it ideal for biometric template generation in privacy-preserving systems.
"""

import secrets
from typing import Tuple, Optional, Dict, Any
import numpy as np
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
import structlog

from .constants import (
    ARGON2_TIME_COST,
    ARGON2_MEMORY_COST,
    ARGON2_PARALLELISM,
    ARGON2_HASH_LENGTH,
    ARGON2_SALT_LENGTH,
)
from .exceptions import TemplateGenerationError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class SecureTemplateGenerator:
    """
    Secure biometric template generator using Argon2 hashing.

    This class provides cryptographically secure template generation from
    fused biometric feature vectors. The generated templates are designed
    to be irreversible while preserving uniqueness for authentication.

    Parameters
    ----------
    time_cost : int, default=ARGON2_TIME_COST
        Number of iterations for Argon2 hashing.
    memory_cost : int, default=ARGON2_MEMORY_COST
        Memory usage in KB for Argon2 hashing.
    parallelism : int, default=ARGON2_PARALLELISM
        Number of parallel threads for Argon2.
    hash_length : int, default=ARGON2_HASH_LENGTH
        Length of the output hash in bytes.
    salt_length : int, default=ARGON2_SALT_LENGTH
        Length of the cryptographic salt in bytes.

    Examples
    --------
    >>> generator = SecureTemplateGenerator()
    >>> fused_vector = np.random.randn(1024)
    >>> template_hash, salt = generator.generate_secure_template(fused_vector)
    >>> print(f"Template hash: {template_hash[:16]}...")
    """

    def __init__(
        self,
        time_cost: int = ARGON2_TIME_COST,
        memory_cost: int = ARGON2_MEMORY_COST,
        parallelism: int = ARGON2_PARALLELISM,
        hash_length: int = ARGON2_HASH_LENGTH,
        salt_length: int = ARGON2_SALT_LENGTH,
    ) -> None:
        self.time_cost = time_cost
        self.memory_cost = memory_cost
        self.parallelism = parallelism
        self.hash_length = hash_length
        self.salt_length = salt_length

        # Validate parameters
        self._validate_parameters()

        # Initialize Argon2 password hasher with custom parameters
        try:
            self.hasher = PasswordHasher(
                time_cost=self.time_cost,
                memory_cost=self.memory_cost,
                parallelism=self.parallelism,
                hash_len=self.hash_length,
                salt_len=self.salt_length,
            )
        except Exception as e:
            raise TemplateGenerationError(
                f"Failed to initialize Argon2 hasher: {str(e)}", algorithm="argon2"
            )

        logger.info(
            "SecureTemplateGenerator initialized",
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_length=hash_length,
            salt_length=salt_length,
        )

    def _validate_parameters(self) -> None:
        """
        Validate Argon2 parameters for security and feasibility.

        Raises
        ------
        TemplateGenerationError
            If any parameters are invalid.
        """
        if self.time_cost < 1:
            raise TemplateGenerationError(
                f"time_cost must be at least 1, got {self.time_cost}",
                algorithm="argon2",
            )

        if self.memory_cost < 8:  # Minimum 8 KB
            raise TemplateGenerationError(
                f"memory_cost must be at least 8 KB, got {self.memory_cost}",
                algorithm="argon2",
            )

        if self.parallelism < 1:
            raise TemplateGenerationError(
                f"parallelism must be at least 1, got {self.parallelism}",
                algorithm="argon2",
            )

        if self.hash_length < 16:  # Minimum 128 bits
            raise TemplateGenerationError(
                f"hash_length must be at least 16 bytes, got {self.hash_length}",
                algorithm="argon2",
            )

        if self.salt_length < 8:  # Minimum 64 bits
            raise TemplateGenerationError(
                f"salt_length must be at least 8 bytes, got {self.salt_length}",
                algorithm="argon2",
            )

    def generate_cryptographic_salt(self) -> bytes:
        """
        Generate a cryptographically secure random salt.

        Returns
        -------
        bytes
            Cryptographically secure random salt.

        Examples
        --------
        >>> generator = SecureTemplateGenerator()
        >>> salt = generator.generate_cryptographic_salt()
        >>> assert len(salt) == ARGON2_SALT_LENGTH
        """
        try:
            salt = secrets.token_bytes(self.salt_length)
            logger.debug(
                "Cryptographic salt generated",
                salt_length=len(salt),
                salt_hex=salt.hex()[:16] + "...",  # Log first 8 bytes for debugging
            )
            return salt
        except Exception as e:
            raise TemplateGenerationError(
                f"Failed to generate cryptographic salt: {str(e)}", algorithm="argon2"
            )

    def _prepare_vector_for_hashing(self, fused_vector: np.ndarray) -> bytes:
        """
        Prepare fused vector for hashing by converting to bytes.

        Parameters
        ----------
        fused_vector : np.ndarray
            Fused biometric feature vector.

        Returns
        -------
        bytes
            Vector data as bytes suitable for hashing.

        Raises
        ------
        TemplateGenerationError
            If vector preparation fails.
        """
        if not isinstance(fused_vector, np.ndarray):
            raise TemplateGenerationError(
                f"fused_vector must be numpy array, got {type(fused_vector)}",
                algorithm="argon2",
            )

        if fused_vector.size == 0:
            raise TemplateGenerationError(
                "fused_vector cannot be empty", algorithm="argon2"
            )

        if not np.isfinite(fused_vector).all():
            raise TemplateGenerationError(
                "fused_vector contains non-finite values", algorithm="argon2"
            )

        try:
            # Convert to consistent float32 format for deterministic results
            normalized_vector = fused_vector.astype(np.float32)

            # Convert to bytes using tobytes() for consistency
            vector_bytes = normalized_vector.tobytes()

            logger.debug(
                "Vector prepared for hashing",
                original_shape=fused_vector.shape,
                original_dtype=str(fused_vector.dtype),
                bytes_length=len(vector_bytes),
            )

            return vector_bytes

        except Exception as e:
            raise TemplateGenerationError(
                f"Failed to prepare vector for hashing: {str(e)}", algorithm="argon2"
            )

    def generate_secure_template(
        self, fused_vector: np.ndarray, custom_salt: Optional[bytes] = None
    ) -> Tuple[str, bytes]:
        """
        Generate a cryptographically secure template from fused biometric vector.

        This method applies Argon2 hashing to the fused biometric feature vector,
        creating an irreversible but deterministic template suitable for
        zero-knowledge proofs and secure storage.

        Parameters
        ----------
        fused_vector : np.ndarray
            Fused biometric feature vector from multimodal fusion.
        custom_salt : Optional[bytes], default=None
            Custom salt for deterministic testing. If None, generates random salt.

        Returns
        -------
        Tuple[str, bytes]
            Tuple containing:
            - template_hash: Hexadecimal string of the Argon2 hash
            - salt: Cryptographic salt used for hashing

        Raises
        ------
        TemplateGenerationError
            If template generation fails for any reason.

        Examples
        --------
        >>> generator = SecureTemplateGenerator()
        >>> vector = np.random.randn(1024)
        >>> hash_str, salt = generator.generate_secure_template(vector)
        >>> assert len(hash_str) == 64  # 32 bytes * 2 hex chars
        >>> assert len(salt) == 16
        """
        logger.info(
            "Starting secure template generation",
            vector_shape=fused_vector.shape,
            vector_dtype=str(fused_vector.dtype),
            custom_salt_provided=custom_salt is not None,
        )

        try:
            # Prepare vector for hashing
            vector_bytes = self._prepare_vector_for_hashing(fused_vector)

            # Generate or use provided salt
            if custom_salt is not None:
                if len(custom_salt) != self.salt_length:
                    raise TemplateGenerationError(
                        f"Custom salt must be {self.salt_length} bytes, got {len(custom_salt)}",
                        algorithm="argon2",
                    )
                salt = custom_salt
                logger.debug("Using provided custom salt")
            else:
                salt = self.generate_cryptographic_salt()

            # Apply Argon2 hashing
            try:
                # Convert vector bytes to string for Argon2 (it expects password as str)
                # We use the hex representation to ensure valid string encoding
                vector_hex_str = vector_bytes.hex()

                # Generate hash using Argon2
                hash_result = self.hasher.hash(vector_hex_str, salt=salt)

                # Extract just the hash part (Argon2 returns full encoded string)
                # The hash is the last part after the final '$'
                hash_parts = hash_result.split("$")
                if len(hash_parts) < 6:
                    raise TemplateGenerationError(
                        "Invalid Argon2 hash format returned", algorithm="argon2"
                    )

                # The hash is base64 encoded, we need to decode and re-encode as hex
                import base64

                hash_b64 = hash_parts[-1]
                hash_bytes = base64.b64decode(hash_b64 + "==")  # Add padding if needed
                template_hash = hash_bytes.hex()

            except Argon2Error as e:
                raise TemplateGenerationError(
                    f"Argon2 hashing failed: {str(e)}", algorithm="argon2"
                )
            except Exception as e:
                raise TemplateGenerationError(
                    f"Unexpected error during hashing: {str(e)}", algorithm="argon2"
                )

            # Validate output
            if len(template_hash) != self.hash_length * 2:  # hex is 2 chars per byte
                raise TemplateGenerationError(
                    f"Generated hash has wrong length: {len(template_hash)}, expected {self.hash_length * 2}",
                    algorithm="argon2",
                )

            logger.info(
                "Secure template generation completed",
                template_hash_length=len(template_hash),
                salt_length=len(salt),
                template_hash_preview=template_hash[:16] + "...",
                vector_stats={
                    "shape": fused_vector.shape,
                    "mean": float(np.mean(fused_vector)),
                    "std": float(np.std(fused_vector)),
                },
            )

            return template_hash, salt

        except Exception as e:
            if isinstance(e, TemplateGenerationError):
                raise
            else:
                raise TemplateGenerationError(
                    f"Unexpected error during template generation: {str(e)}",
                    algorithm="argon2",
                )

    def verify_template(
        self, fused_vector: np.ndarray, template_hash: str, salt: bytes
    ) -> bool:
        """
        Verify that a fused vector produces the expected template hash.

        Parameters
        ----------
        fused_vector : np.ndarray
            Fused biometric feature vector to verify.
        template_hash : str
            Expected template hash as hexadecimal string.
        salt : bytes
            Salt used in original template generation.

        Returns
        -------
        bool
            True if verification succeeds, False otherwise.

        Examples
        --------
        >>> generator = SecureTemplateGenerator()
        >>> vector = np.random.randn(1024)
        >>> hash_str, salt = generator.generate_secure_template(vector)
        >>> is_valid = generator.verify_template(vector, hash_str, salt)
        >>> assert is_valid
        """
        logger.debug(
            "Starting template verification",
            template_hash_preview=template_hash[:16] + "...",
            salt_length=len(salt),
        )

        try:
            # Generate template with the provided salt
            computed_hash, _ = self.generate_secure_template(
                fused_vector, custom_salt=salt
            )

            # Compare hashes
            is_valid = computed_hash == template_hash

            logger.info(
                "Template verification completed",
                verification_result=is_valid,
                computed_hash_preview=computed_hash[:16] + "...",
                expected_hash_preview=template_hash[:16] + "...",
            )

            return is_valid

        except Exception as e:
            logger.error(
                "Template verification failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def get_template_metadata(self, template_hash: str, salt: bytes) -> Dict[str, Any]:
        """
        Get metadata about a generated template.

        Parameters
        ----------
        template_hash : str
            Template hash as hexadecimal string.
        salt : bytes
            Salt used in template generation.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing template information.
        """
        return {
            "template_hash": template_hash,
            "template_hash_length": len(template_hash),
            "salt_hex": salt.hex(),
            "salt_length": len(salt),
            "argon2_parameters": {
                "time_cost": self.time_cost,
                "memory_cost": self.memory_cost,
                "parallelism": self.parallelism,
                "hash_length": self.hash_length,
                "salt_length": self.salt_length,
            },
            "algorithm": "argon2",
            "template_version": "1.0",
        }


# Convenience functions for simple usage
def generate_secure_template(fused_vector: np.ndarray) -> Tuple[str, bytes]:
    """
    Convenience function to generate a secure template with default settings.

    Parameters
    ----------
    fused_vector : np.ndarray
        Fused biometric feature vector.

    Returns
    -------
    Tuple[str, bytes]
        Tuple containing template hash and salt.

    Raises
    ------
    TemplateGenerationError
        If template generation fails.

    Examples
    --------
    >>> fused_vector = np.random.randn(1024)
    >>> template_hash, salt = generate_secure_template(fused_vector)
    >>> print(f"Generated template: {template_hash[:16]}...")
    """
    generator = SecureTemplateGenerator()
    return generator.generate_secure_template(fused_vector)


def verify_secure_template(
    fused_vector: np.ndarray, template_hash: str, salt: bytes
) -> bool:
    """
    Convenience function to verify a secure template.

    Parameters
    ----------
    fused_vector : np.ndarray
        Fused biometric feature vector.
    template_hash : str
        Expected template hash.
    salt : bytes
        Salt used in original generation.

    Returns
    -------
    bool
        True if verification succeeds.

    Examples
    --------
    >>> is_valid = verify_secure_template(vector, hash_str, salt)
    >>> assert is_valid
    """
    generator = SecureTemplateGenerator()
    return generator.verify_template(fused_vector, template_hash, salt)


def benchmark_template_generation(
    vector_size: int = 1024, n_iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark template generation performance.

    Parameters
    ----------
    vector_size : int, default=1024
        Size of test vectors to use.
    n_iterations : int, default=100
        Number of iterations to run.

    Returns
    -------
    Dict[str, Any]
        Benchmark results including timing statistics.

    Examples
    --------
    >>> results = benchmark_template_generation(1024, 50)
    >>> print(f"Average time: {results['avg_time_ms']:.2f} ms")
    """
    import time

    logger.info(
        "Starting template generation benchmark",
        vector_size=vector_size,
        n_iterations=n_iterations,
    )

    generator = SecureTemplateGenerator()
    times = []

    for i in range(n_iterations):
        # Generate random vector
        test_vector = np.random.randn(vector_size).astype(np.float32)

        # Time the template generation
        start_time = time.time()
        template_hash, salt = generator.generate_secure_template(test_vector)
        end_time = time.time()

        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    results = {
        "n_iterations": n_iterations,
        "vector_size": vector_size,
        "avg_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "median_time_ms": np.median(times),
        "total_time_ms": np.sum(times),
    }

    logger.info("Template generation benchmark completed", **results)

    return results
