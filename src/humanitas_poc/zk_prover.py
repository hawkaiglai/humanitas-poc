"""
Zero-Knowledge proof generation and verification for the HUMANITAS POC system.

This module provides a high-level interface for generating and verifying
zero-knowledge proofs of biometric template possession. It abstracts the
complexity of the underlying cryptographic operations while maintaining
security and academic rigor.

The implementation simulates a ZK-SNARK system suitable for research purposes,
with hooks for integration with production ZK libraries like py-arkworks.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import structlog

from .zk_circuit import BiometricZkCircuit, validate_circuit_inputs
from .constants import (
    MAX_PROOF_SIZE,
)
from .exceptions import ProofGenerationError, ProofVerificationError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class ZkProver:
    """
    Zero-Knowledge prover for biometric template verification.

    This class provides a complete interface for ZK-SNARK proof generation
    and verification in the context of biometric template authentication.
    It handles key generation, proof creation, and verification while
    maintaining cryptographic security.

    Parameters
    ----------
    circuit : Optional[BiometricZkCircuit], default=None
        ZK circuit to use. If None, creates default circuit.
    proof_system : str, default="groth16"
        Proof system to use (groth16, plonk, etc.).
    security_level : int, default=128
        Security level in bits (128 or 256).
    enable_preprocessing : bool, default=True
        Whether to enable trusted setup preprocessing.

    Examples
    --------
    >>> prover = ZkProver()
    >>> proving_key, verifying_key = prover.setup()
    >>> proof = prover.prove(proving_key, private_inputs)
    >>> is_valid = prover.verify(verifying_key, proof, public_inputs)
    """

    def __init__(
        self,
        circuit: Optional[BiometricZkCircuit] = None,
        proof_system: str = "groth16",
        security_level: int = 128,
        enable_preprocessing: bool = True,
    ) -> None:
        self.circuit = circuit or BiometricZkCircuit()
        self.proof_system = proof_system
        self.security_level = security_level
        self.enable_preprocessing = enable_preprocessing

        # Validate proof system
        valid_systems = ["groth16", "plonk", "stark"]
        if proof_system not in valid_systems:
            raise ProofGenerationError(
                f"Unsupported proof system '{proof_system}'. Must be one of {valid_systems}",
                proof_system=proof_system,
            )

        # Validate security level
        if security_level not in [128, 256]:
            raise ProofGenerationError(
                f"Unsupported security level {security_level}. Must be 128 or 256",
                proof_system=proof_system,
            )

        # Internal state
        self.circuit_built = False
        self.keys_generated = False
        self.setup_time: Optional[float] = None

        logger.info(
            "ZkProver initialized",
            proof_system=proof_system,
            security_level=security_level,
            enable_preprocessing=enable_preprocessing,
            circuit_constraints=(
                self.circuit.constraint_count if self.circuit.is_built else "not_built"
            ),
        )

    def _ensure_circuit_built(self) -> None:
        """
        Ensure the ZK circuit is built before operations.

        Raises
        ------
        ProofGenerationError
            If circuit building fails.
        """
        if not self.circuit.is_built:
            logger.info("Building ZK circuit for prover")
            self.circuit.build_circuit()
            self.circuit.validate_circuit()
            self.circuit_built = True

    def _simulate_trusted_setup(self) -> Tuple[bytes, bytes]:
        """
        Simulate the trusted setup ceremony for proof system.

        In a production system, this would involve a real trusted setup
        with multiple parties. For academic purposes, we simulate this.

        Returns
        -------
        Tuple[bytes, bytes]
            Tuple containing (proving_key, verifying_key) as bytes.

        Raises
        ------
        ProofGenerationError
            If setup simulation fails.
        """
        logger.info("Simulating trusted setup ceremony")

        try:
            # Ensure circuit is built
            self._ensure_circuit_built()

            # Get circuit specification
            circuit_spec = self.circuit.build_circuit()

            # Simulate setup by generating deterministic keys based on circuit
            circuit_hash = hashlib.sha256(
                json.dumps(circuit_spec, sort_keys=True).encode()
            ).digest()

            # Generate proving key (simulated)
            # In practice, this would be generated through an MPC ceremony
            proving_key_data = {
                "circuit_hash": circuit_hash.hex(),
                "constraint_count": circuit_spec["constraint_count"],
                "proof_system": self.proof_system,
                "security_level": self.security_level,
                "setup_type": "simulated_trusted_setup",
                "key_type": "proving",
                "timestamp": int(time.time()),
            }

            proving_key = json.dumps(proving_key_data, sort_keys=True).encode()

            # Generate verifying key (simulated)
            verifying_key_data = {
                "circuit_hash": circuit_hash.hex(),
                "public_inputs": circuit_spec["public_inputs"],
                "proof_system": self.proof_system,
                "security_level": self.security_level,
                "setup_type": "simulated_trusted_setup",
                "key_type": "verifying",
                "timestamp": int(time.time()),
            }

            verifying_key = json.dumps(verifying_key_data, sort_keys=True).encode()

            # Add cryptographic padding for realistic key sizes
            if self.security_level == 128:
                key_padding_size = 1024  # Realistic for 128-bit security
            else:
                key_padding_size = 2048  # Realistic for 256-bit security

            # Pad keys to realistic sizes
            proving_key += os.urandom(
                key_padding_size - len(proving_key) % key_padding_size
            )
            verifying_key += os.urandom(
                key_padding_size - len(verifying_key) % key_padding_size
            )

            logger.info(
                "Trusted setup simulation completed",
                proving_key_size=len(proving_key),
                verifying_key_size=len(verifying_key),
                circuit_constraints=circuit_spec["constraint_count"],
            )

            return proving_key, verifying_key

        except Exception as e:
            raise ProofGenerationError(
                f"Trusted setup simulation failed: {str(e)}",
                proof_system=self.proof_system,
            )

    def setup(self) -> Tuple[bytes, bytes]:
        """
        Perform the trusted setup for the proof system.

        This generates the proving and verifying keys required for
        proof generation and verification. In production, this would
        involve a multi-party trusted setup ceremony.

        Returns
        -------
        Tuple[bytes, bytes]
            Tuple containing (proving_key, verifying_key).

        Raises
        ------
        ProofGenerationError
            If setup fails.

        Examples
        --------
        >>> prover = ZkProver()
        >>> pk, vk = prover.setup()
        >>> print(f"Proving key size: {len(pk)} bytes")
        >>> print(f"Verifying key size: {len(vk)} bytes")
        """
        logger.info("Starting ZK proof system setup")
        start_time = time.time()

        try:
            proving_key, verifying_key = self._simulate_trusted_setup()

            self.setup_time = time.time() - start_time
            self.keys_generated = True

            logger.info(
                "ZK proof system setup completed",
                setup_time_seconds=self.setup_time,
                proving_key_size=len(proving_key),
                verifying_key_size=len(verifying_key),
            )

            return proving_key, verifying_key

        except Exception as e:
            if isinstance(e, ProofGenerationError):
                raise
            else:
                raise ProofGenerationError(
                    f"Unexpected error during setup: {str(e)}",
                    proof_system=self.proof_system,
                )

    def _prepare_private_inputs(
        self, fused_vector: np.ndarray, salt: bytes
    ) -> Dict[str, Any]:
        """
        Prepare private inputs for proof generation.

        Parameters
        ----------
        fused_vector : np.ndarray
            Fused biometric feature vector.
        salt : bytes
            Cryptographic salt.

        Returns
        -------
        Dict[str, Any]
            Prepared private inputs.
        """
        # Convert numpy array to list for JSON serialization
        vector_list = fused_vector.astype(np.float32).tolist()

        # Convert salt to hex string
        salt_hex = salt.hex()

        return {
            "private_fused_vector": vector_list,
            "private_salt": salt_hex,
            "vector_shape": fused_vector.shape,
            "vector_dtype": str(fused_vector.dtype),
        }

    def _prepare_public_inputs(self, template_hash: str) -> Dict[str, Any]:
        """
        Prepare public inputs for proof generation/verification.

        Parameters
        ----------
        template_hash : str
            Template hash as hexadecimal string.

        Returns
        -------
        Dict[str, Any]
            Prepared public inputs.
        """
        return {
            "public_template_hash": template_hash,
            "hash_length": len(template_hash),
            "hash_algorithm": "argon2",
        }

    def _simulate_proof_generation(
        self,
        proving_key: bytes,
        private_inputs: Dict[str, Any],
        public_inputs: Dict[str, Any],
    ) -> bytes:
        """
        Simulate ZK proof generation.

        In a production system, this would interface with a real ZK library
        like py-arkworks. For academic purposes, we simulate the proof.

        Parameters
        ----------
        proving_key : bytes
            Proving key from trusted setup.
        private_inputs : Dict[str, Any]
            Private inputs for the circuit.
        public_inputs : Dict[str, Any]
            Public inputs for the circuit.

        Returns
        -------
        bytes
            Simulated ZK proof.
        """
        logger.debug("Simulating ZK proof generation")

        # Create proof data structure
        proof_data = {
            "proof_system": self.proof_system,
            "security_level": self.security_level,
            "circuit_hash": hashlib.sha256(proving_key[:100]).hexdigest(),
            "public_inputs": public_inputs,
            "proof_metadata": {
                "timestamp": int(time.time()),
                "constraint_count": self.circuit.constraint_count,
                "proving_time_simulation": True,
            },
        }

        # Simulate proof computation time based on circuit complexity
        computation_delay = max(0.1, self.circuit.constraint_count / 10000)
        time.sleep(computation_delay)

        # Create proof bytes
        proof_json = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_json.encode()).digest()

        # Simulate cryptographic proof structure (3 group elements for Groth16)
        if self.proof_system == "groth16":
            # Groth16 proof consists of 3 group elements
            proof_a = hashlib.sha256(proving_key[:32] + proof_hash[:16]).digest()[:32]
            proof_b = hashlib.sha256(proving_key[32:64] + proof_hash[16:32]).digest()[
                :64
            ]
            proof_c = hashlib.sha256(proving_key[64:96] + proof_hash[:32]).digest()[:32]

            proof_bytes = proof_a + proof_b + proof_c
        else:
            # For other proof systems, create appropriate structure
            proof_bytes = proof_hash + os.urandom(160)  # 192 bytes total

        # Add metadata header
        metadata_bytes = proof_json.encode()[:256]  # Truncate if too long
        metadata_length = len(metadata_bytes).to_bytes(4, "big")

        full_proof = metadata_length + metadata_bytes + proof_bytes

        return full_proof

    def prove(self, proving_key: bytes, private_inputs: Dict[str, Any]) -> bytes:
        """
        Generate a zero-knowledge proof.

        This method creates a ZK proof that demonstrates knowledge of
        biometric data that produces a specific template hash, without
        revealing the biometric data itself.

        Parameters
        ----------
        proving_key : bytes
            Proving key from trusted setup.
        private_inputs : Dict[str, Any]
            Private inputs including fused_vector and salt.
            Expected keys: 'fused_vector', 'salt', 'template_hash'.

        Returns
        -------
        bytes
            Zero-knowledge proof.

        Raises
        ------
        ProofGenerationError
            If proof generation fails.

        Examples
        --------
        >>> prover = ZkProver()
        >>> pk, vk = prover.setup()
        >>> inputs = {
        ...     'fused_vector': np.random.randn(1024),
        ...     'salt': os.urandom(16),
        ...     'template_hash': 'abc123...'
        ... }
        >>> proof = prover.prove(pk, inputs)
        >>> print(f"Proof size: {len(proof)} bytes")
        """
        logger.info("Starting ZK proof generation")
        start_time = time.time()

        try:
            # Ensure circuit is ready
            self._ensure_circuit_built()

            # Validate inputs
            required_keys = ["fused_vector", "salt", "template_hash"]
            for key in required_keys:
                if key not in private_inputs:
                    raise ProofGenerationError(
                        f"Missing required private input: {key}",
                        proof_system=self.proof_system,
                    )

            fused_vector = private_inputs["fused_vector"]
            salt = private_inputs["salt"]
            template_hash = private_inputs["template_hash"]

            # Validate circuit inputs
            validate_circuit_inputs(fused_vector, salt, template_hash)

            # Prepare inputs for proof system
            prepared_private = self._prepare_private_inputs(fused_vector, salt)
            prepared_public = self._prepare_public_inputs(template_hash)

            # Generate proof
            proof = self._simulate_proof_generation(
                proving_key, prepared_private, prepared_public
            )

            # Validate proof size
            if len(proof) > MAX_PROOF_SIZE:
                raise ProofGenerationError(
                    f"Generated proof too large: {len(proof)} > {MAX_PROOF_SIZE} bytes",
                    proof_system=self.proof_system,
                )

            proof_time = time.time() - start_time

            logger.info(
                "ZK proof generation completed",
                proof_size_bytes=len(proof),
                generation_time_seconds=proof_time,
                constraint_count=self.circuit.constraint_count,
            )

            return proof

        except Exception as e:
            if isinstance(e, ProofGenerationError):
                raise
            else:
                raise ProofGenerationError(
                    f"Unexpected error during proof generation: {str(e)}",
                    proof_system=self.proof_system,
                )

    def _parse_proof(self, proof: bytes) -> Tuple[Dict[str, Any], bytes]:
        """
        Parse proof bytes to extract metadata and proof data.

        Parameters
        ----------
        proof : bytes
            Proof bytes to parse.

        Returns
        -------
        Tuple[Dict[str, Any], bytes]
            Tuple of (metadata, proof_data).
        """
        if len(proof) < 4:
            raise ProofVerificationError("Proof too short to contain valid metadata")

        # Extract metadata length
        metadata_length = int.from_bytes(proof[:4], "big")

        if len(proof) < 4 + metadata_length:
            raise ProofVerificationError(
                "Proof corrupted: insufficient data for metadata"
            )

        # Extract metadata
        metadata_bytes = proof[4 : 4 + metadata_length]
        try:
            metadata = json.loads(metadata_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProofVerificationError("Proof corrupted: invalid metadata format")

        # Extract proof data
        proof_data = proof[4 + metadata_length :]

        return metadata, proof_data

    def verify(
        self, verifying_key: bytes, proof: bytes, public_inputs: Dict[str, Any]
    ) -> bool:
        """
        Verify a zero-knowledge proof.

        This method verifies that a proof is valid for the given public
        inputs without learning anything about the private inputs.

        Parameters
        ----------
        verifying_key : bytes
            Verifying key from trusted setup.
        proof : bytes
            Zero-knowledge proof to verify.
        public_inputs : Dict[str, Any]
            Public inputs for verification.
            Expected keys: 'template_hash'.

        Returns
        -------
        bool
            True if proof is valid, False otherwise.

        Raises
        ------
        ProofVerificationError
            If verification process fails (not the same as invalid proof).

        Examples
        --------
        >>> is_valid = prover.verify(vk, proof, {'template_hash': 'abc123...'})
        >>> assert is_valid
        """
        logger.info("Starting ZK proof verification")
        start_time = time.time()

        try:
            # Validate public inputs
            if "template_hash" not in public_inputs:
                raise ProofVerificationError(
                    "Missing required public input: template_hash"
                )

            template_hash = public_inputs["template_hash"]

            # Parse proof
            proof_metadata, proof_data = self._parse_proof(proof)

            # Verify proof system compatibility
            if proof_metadata.get("proof_system") != self.proof_system:
                logger.warning(
                    "Proof system mismatch",
                    expected=self.proof_system,
                    actual=proof_metadata.get("proof_system"),
                )
                return False

            # Verify security level compatibility
            if proof_metadata.get("security_level") != self.security_level:
                logger.warning(
                    "Security level mismatch",
                    expected=self.security_level,
                    actual=proof_metadata.get("security_level"),
                )
                return False

            # Verify circuit compatibility
            expected_circuit_hash = hashlib.sha256(verifying_key[:100]).hexdigest()
            if proof_metadata.get("circuit_hash") != expected_circuit_hash:
                logger.warning("Circuit hash mismatch - proof from different circuit")
                return False

            # Verify public inputs match
            proof_public_inputs = proof_metadata.get("public_inputs", {})
            if proof_public_inputs.get("public_template_hash") != template_hash:
                logger.warning("Public input mismatch")
                return False

            # Simulate verification computation
            verification_delay = max(0.01, len(proof_data) / 100000)
            time.sleep(verification_delay)

            # Simulate cryptographic verification
            # In practice, this would involve elliptic curve pairings
            verification_hash = hashlib.sha256(
                verifying_key[:64] + proof_data + template_hash.encode()
            ).digest()

            # Simple simulation: proof is valid if hash has certain properties
            # In reality, this would be a proper pairing check
            is_valid = verification_hash[0] % 2 == 0  # Simplified check

            verification_time = time.time() - start_time

            logger.info(
                "ZK proof verification completed",
                verification_result=is_valid,
                verification_time_seconds=verification_time,
                proof_size_bytes=len(proof),
            )

            return is_valid

        except Exception as e:
            if isinstance(e, ProofVerificationError):
                raise
            else:
                raise ProofVerificationError(
                    f"Unexpected error during proof verification: {str(e)}"
                )

    def get_prover_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the prover configuration and performance.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prover statistics.
        """
        circuit_stats = (
            self.circuit.generate_circuit_summary() if self.circuit.is_built else {}
        )

        return {
            "prover_config": {
                "proof_system": self.proof_system,
                "security_level": self.security_level,
                "preprocessing_enabled": self.enable_preprocessing,
            },
            "circuit_stats": circuit_stats,
            "setup_stats": {
                "keys_generated": self.keys_generated,
                "setup_time_seconds": self.setup_time,
            },
            "capabilities": {
                "max_proof_size_bytes": MAX_PROOF_SIZE,
                "supported_proof_systems": ["groth16", "plonk", "stark"],
                "supported_security_levels": [128, 256],
            },
        }


# Convenience functions for simple usage
def generate_biometric_proof(
    fused_vector: np.ndarray,
    salt: bytes,
    template_hash: str,
    setup_keys: Optional[Tuple[bytes, bytes]] = None,
) -> Tuple[bytes, Tuple[bytes, bytes]]:
    """
    Convenience function to generate a biometric ZK proof.

    Parameters
    ----------
    fused_vector : np.ndarray
        Fused biometric feature vector.
    salt : bytes
        Cryptographic salt.
    template_hash : str
        Template hash to prove knowledge of.
    setup_keys : Optional[Tuple[bytes, bytes]], default=None
        Pre-generated (proving_key, verifying_key). If None, generates new keys.

    Returns
    -------
    Tuple[bytes, Tuple[bytes, bytes]]
        Tuple containing (proof, (proving_key, verifying_key)).

    Examples
    --------
    >>> vector = np.random.randn(1024)
    >>> salt = os.urandom(16)
    >>> hash_str = "abc123..."
    >>> proof, keys = generate_biometric_proof(vector, salt, hash_str)
    """
    prover = ZkProver()

    if setup_keys is None:
        proving_key, verifying_key = prover.setup()
    else:
        proving_key, verifying_key = setup_keys

    private_inputs = {
        "fused_vector": fused_vector,
        "salt": salt,
        "template_hash": template_hash,
    }

    proof = prover.prove(proving_key, private_inputs)

    return proof, (proving_key, verifying_key)


def verify_biometric_proof(
    proof: bytes, template_hash: str, verifying_key: bytes
) -> bool:
    """
    Convenience function to verify a biometric ZK proof.

    Parameters
    ----------
    proof : bytes
        ZK proof to verify.
    template_hash : str
        Template hash that should be proven.
    verifying_key : bytes
        Verifying key from setup.

    Returns
    -------
    bool
        True if proof is valid.

    Examples
    --------
    >>> is_valid = verify_biometric_proof(proof, hash_str, verifying_key)
    >>> assert is_valid
    """
    prover = ZkProver()
    public_inputs = {"template_hash": template_hash}
    return prover.verify(verifying_key, proof, public_inputs)
