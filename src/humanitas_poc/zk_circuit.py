"""
Zero-Knowledge circuit definition for the HUMANITAS POC system.

This module defines the ZK-SNARK circuit structure for proving knowledge of
biometric templates without revealing the underlying biometric data. The circuit
implements Argon2 hashing verification within the zero-knowledge framework.

The circuit enables privacy-preserving biometric authentication where users can
prove possession of valid biometric credentials without exposing the actual
biometric features or templates.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import structlog

from .constants import (
    ARGON2_TIME_COST,
    ARGON2_MEMORY_COST,
    ARGON2_HASH_LENGTH,
    ZK_CONSTRAINT_SYSTEM_SIZE,
)
from .exceptions import ProofGenerationError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class BiometricZkCircuit:
    """
    Zero-Knowledge circuit for biometric template verification.

    This class defines the constraint system for proving knowledge of a biometric
    template that hashes to a specific public value using Argon2, without
    revealing the template itself.

    The circuit implements:
    1. Argon2 hash computation within ZK constraints
    2. Equality assertion between computed and expected hash
    3. Range proofs for input validity
    4. Privacy-preserving template verification

    Parameters
    ----------
    constraint_system_size : int, default=ZK_CONSTRAINT_SYSTEM_SIZE
        Size of the constraint system (affects proof size and time).
    argon2_time_cost : int, default=ARGON2_TIME_COST
        Argon2 time cost parameter to verify.
    argon2_memory_cost : int, default=ARGON2_MEMORY_COST
        Argon2 memory cost parameter to verify.
    enable_optimizations : bool, default=True
        Whether to enable circuit optimizations.

    Examples
    --------
    >>> circuit = BiometricZkCircuit()
    >>> constraints = circuit.build_circuit()
    >>> print(f"Circuit has {len(constraints)} constraints")
    """

    def __init__(
        self,
        constraint_system_size: int = ZK_CONSTRAINT_SYSTEM_SIZE,
        argon2_time_cost: int = ARGON2_TIME_COST,
        argon2_memory_cost: int = ARGON2_MEMORY_COST,
        enable_optimizations: bool = True,
    ) -> None:
        self.constraint_system_size = constraint_system_size
        self.argon2_time_cost = argon2_time_cost
        self.argon2_memory_cost = argon2_memory_cost
        self.enable_optimizations = enable_optimizations

        # Circuit components
        self.circuit_constraints: List[Dict[str, Any]] = []
        self.public_inputs: List[str] = []
        self.private_inputs: List[str] = []
        self.intermediate_variables: Dict[str, Any] = {}

        # Circuit state
        self.is_built = False
        self.constraint_count = 0

        logger.info(
            "BiometricZkCircuit initialized",
            constraint_system_size=constraint_system_size,
            argon2_time_cost=argon2_time_cost,
            argon2_memory_cost=argon2_memory_cost,
            enable_optimizations=enable_optimizations,
        )

    def _add_constraint(
        self,
        constraint_type: str,
        inputs: List[str],
        outputs: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a constraint to the circuit.

        Parameters
        ----------
        constraint_type : str
            Type of constraint (e.g., 'add', 'mul', 'hash', 'assert_eq').
        inputs : List[str]
            Input variable names for the constraint.
        outputs : List[str]
            Output variable names for the constraint.
        parameters : Optional[Dict[str, Any]], default=None
            Additional parameters for the constraint.

        Returns
        -------
        str
            Unique identifier for the added constraint.
        """
        constraint_id = f"constraint_{self.constraint_count}"

        constraint = {
            "id": constraint_id,
            "type": constraint_type,
            "inputs": inputs,
            "outputs": outputs,
            "parameters": parameters or {},
        }

        self.circuit_constraints.append(constraint)
        self.constraint_count += 1

        logger.debug(
            "Constraint added to circuit",
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            inputs=inputs,
            outputs=outputs,
        )

        return constraint_id

    def _create_variable(self, name: str, var_type: str = "intermediate") -> str:
        """
        Create a new variable in the circuit.

        Parameters
        ----------
        name : str
            Variable name.
        var_type : str, default="intermediate"
            Variable type: "public", "private", or "intermediate".

        Returns
        -------
        str
            Full variable name with type prefix.
        """
        full_name = f"{var_type}_{name}"

        if var_type == "public":
            self.public_inputs.append(full_name)
        elif var_type == "private":
            self.private_inputs.append(full_name)
        else:
            self.intermediate_variables[full_name] = None

        return full_name

    def _add_range_proof(self, variable: str, min_value: int, max_value: int) -> str:
        """
        Add range proof constraints for a variable.

        Parameters
        ----------
        variable : str
            Variable name to apply range proof to.
        min_value : int
            Minimum allowed value.
        max_value : int
            Maximum allowed value.

        Returns
        -------
        str
            Constraint ID for the range proof.
        """
        return self._add_constraint(
            "range_proof",
            [variable],
            [],
            {"min_value": min_value, "max_value": max_value},
        )

    def _simulate_argon2_in_circuit(
        self, input_var: str, salt_var: str, output_var: str
    ) -> List[str]:
        """
        Simulate Argon2 hashing within the ZK circuit.

        This creates a series of constraints that verify Argon2 computation
        without implementing the full algorithm in constraints (which would
        be computationally prohibitive).

        Parameters
        ----------
        input_var : str
            Input data variable name.
        salt_var : str
            Salt variable name.
        output_var : str
            Output hash variable name.

        Returns
        -------
        List[str]
            List of constraint IDs for the Argon2 simulation.
        """
        logger.debug(
            "Simulating Argon2 in circuit",
            input_var=input_var,
            salt_var=salt_var,
            output_var=output_var,
        )

        constraint_ids = []

        # For academic purposes, we simulate Argon2 with a series of
        # hash functions and constraints that approximate the behavior

        # Step 1: Combine input and salt
        combined_var = self._create_variable("argon2_combined")
        constraint_ids.append(
            self._add_constraint(
                "combine",
                [input_var, salt_var],
                [combined_var],
                {"operation": "concat"},
            )
        )

        # Step 2: Simulate time cost iterations with hash chains
        current_var = combined_var
        for i in range(min(self.argon2_time_cost, 3)):  # Limit for practical ZK
            iteration_var = self._create_variable(f"argon2_iter_{i}")
            constraint_ids.append(
                self._add_constraint(
                    "hash_sha256", [current_var], [iteration_var], {"iteration": i}
                )
            )
            current_var = iteration_var

        # Step 3: Simulate memory-hard operations (simplified)
        memory_var = self._create_variable("argon2_memory")
        constraint_ids.append(
            self._add_constraint(
                "memory_hard_function",
                [current_var],
                [memory_var],
                {"memory_cost": min(self.argon2_memory_cost, 1024)},  # Simplified
            )
        )

        # Step 4: Final hash extraction
        constraint_ids.append(
            self._add_constraint(
                "hash_extract",
                [memory_var],
                [output_var],
                {"output_length": ARGON2_HASH_LENGTH},
            )
        )

        return constraint_ids

    def build_circuit(self) -> Dict[str, Any]:
        """
        Build the complete ZK circuit for biometric template verification.

        This method constructs the full constraint system including:
        - Public input declarations (template hash)
        - Private input declarations (fused vector, salt)
        - Argon2 simulation constraints
        - Hash equality verification
        - Range proofs for security

        Returns
        -------
        Dict[str, Any]
            Complete circuit specification including constraints and variables.

        Raises
        ------
        ProofGenerationError
            If circuit building fails.

        Examples
        --------
        >>> circuit = BiometricZkCircuit()
        >>> spec = circuit.build_circuit()
        >>> print(f"Circuit built with {spec['constraint_count']} constraints")
        """
        logger.info("Building ZK circuit for biometric template verification")

        try:
            # Reset circuit state
            self.circuit_constraints = []
            self.public_inputs = []
            self.private_inputs = []
            self.intermediate_variables = {}
            self.constraint_count = 0

            # Define public inputs (known to verifier)
            public_template_hash = self._create_variable("template_hash", "public")

            # Define private inputs (known only to prover)
            private_fused_vector = self._create_variable("fused_vector", "private")
            private_salt = self._create_variable("salt", "private")

            # Add range proofs for private inputs (security constraints)
            self._add_range_proof(
                private_fused_vector, -1000, 1000
            )  # Reasonable bounds
            self._add_range_proof(private_salt, 0, 2**128 - 1)  # Valid salt range

            # Create intermediate variable for computed hash
            computed_hash = self._create_variable("computed_hash")

            # Add Argon2 simulation constraints
            self._simulate_argon2_in_circuit(
                private_fused_vector, private_salt, computed_hash
            )

            # Add hash equality assertion (core verification)
            self._add_constraint(
                "assert_equal",
                [computed_hash, public_template_hash],
                [],
                {"error_message": "Template hash mismatch"},
            )

            # Add additional security constraints
            if self.enable_optimizations:
                # Add non-malleability constraints
                self._add_constraint(
                    "non_malleability",
                    [private_fused_vector, private_salt],
                    [],
                    {"check_type": "uniqueness"},
                )

                # Add soundness constraints
                self._add_constraint(
                    "soundness_check", [computed_hash], [], {"entropy_threshold": 0.8}
                )

            # Mark circuit as built
            self.is_built = True

            # Prepare circuit specification
            circuit_spec = {
                "constraint_count": self.constraint_count,
                "public_inputs": self.public_inputs,
                "private_inputs": self.private_inputs,
                "intermediate_variables": list(self.intermediate_variables.keys()),
                "constraints": self.circuit_constraints,
                "circuit_parameters": {
                    "constraint_system_size": self.constraint_system_size,
                    "argon2_time_cost": self.argon2_time_cost,
                    "argon2_memory_cost": self.argon2_memory_cost,
                    "optimizations_enabled": self.enable_optimizations,
                },
                "circuit_metadata": {
                    "version": "1.0",
                    "algorithm": "groth16",
                    "curve": "bn254",
                    "field_size": "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                },
            }

            logger.info(
                "ZK circuit built successfully",
                constraint_count=self.constraint_count,
                public_inputs=len(self.public_inputs),
                private_inputs=len(self.private_inputs),
                intermediate_vars=len(self.intermediate_variables),
            )

            return circuit_spec

        except Exception as e:
            raise ProofGenerationError(
                f"Failed to build ZK circuit: {str(e)}",
                circuit_type="biometric_verification",
            )

    def validate_circuit(self) -> bool:
        """
        Validate the built circuit for correctness and security.

        Returns
        -------
        bool
            True if circuit is valid.

        Raises
        ------
        ProofGenerationError
            If circuit validation fails.
        """
        if not self.is_built:
            raise ProofGenerationError(
                "Circuit must be built before validation",
                circuit_type="biometric_verification",
            )

        logger.info("Validating ZK circuit")

        validation_errors = []

        # Check constraint count is within limits
        if self.constraint_count > self.constraint_system_size:
            validation_errors.append(
                f"Too many constraints: {self.constraint_count} > {self.constraint_system_size}"
            )

        # Check for required public inputs
        if "public_template_hash" not in self.public_inputs:
            validation_errors.append("Missing required public input: template_hash")

        # Check for required private inputs
        required_private = ["private_fused_vector", "private_salt"]
        for req_input in required_private:
            if req_input not in self.private_inputs:
                validation_errors.append(f"Missing required private input: {req_input}")

        # Check for circular dependencies in constraints
        # (Simplified check - in practice would need more sophisticated analysis)
        output_vars = set()
        for constraint in self.circuit_constraints:
            for output in constraint["outputs"]:
                if output in output_vars:
                    validation_errors.append(
                        f"Variable {output} defined multiple times"
                    )
                output_vars.add(output)

        if validation_errors:
            error_msg = "Circuit validation failed:\n" + "\n".join(validation_errors)
            raise ProofGenerationError(error_msg, circuit_type="biometric_verification")

        logger.info("ZK circuit validation completed successfully")
        return True

    def estimate_proof_size(self) -> Dict[str, int]:
        """
        Estimate the size of proofs generated by this circuit.

        Returns
        -------
        Dict[str, int]
            Estimated proof sizes in bytes for different components.
        """
        # These are estimates based on typical Groth16 proof sizes
        base_proof_size = 192  # 3 group elements * 64 bytes each

        # Additional size for complex constraints
        constraint_overhead = self.constraint_count * 2

        # Public input size
        public_input_size = len(self.public_inputs) * 32  # 32 bytes per field element

        return {
            "base_proof_bytes": base_proof_size,
            "constraint_overhead_bytes": constraint_overhead,
            "public_input_bytes": public_input_size,
            "total_estimated_bytes": base_proof_size
            + constraint_overhead
            + public_input_size,
            "verification_key_bytes": 256 + (len(self.public_inputs) * 64),
        }

    def generate_circuit_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the circuit.

        Returns
        -------
        Dict[str, Any]
            Circuit summary including statistics and metadata.
        """
        if not self.is_built:
            return {"error": "Circuit not built"}

        constraint_types = {}
        for constraint in self.circuit_constraints:
            ctype = constraint["type"]
            constraint_types[ctype] = constraint_types.get(ctype, 0) + 1

        return {
            "circuit_built": self.is_built,
            "total_constraints": self.constraint_count,
            "constraint_types": constraint_types,
            "public_inputs": len(self.public_inputs),
            "private_inputs": len(self.private_inputs),
            "intermediate_variables": len(self.intermediate_variables),
            "estimated_proof_size": self.estimate_proof_size(),
            "circuit_parameters": {
                "constraint_system_size": self.constraint_system_size,
                "argon2_time_cost": self.argon2_time_cost,
                "argon2_memory_cost": self.argon2_memory_cost,
                "optimizations_enabled": self.enable_optimizations,
            },
            "security_level": (
                "128-bit" if self.constraint_system_size >= 2048 else "80-bit"
            ),
        }


def create_biometric_circuit(
    argon2_params: Optional[Dict[str, int]] = None,
) -> BiometricZkCircuit:
    """
    Convenience function to create a biometric verification circuit.

    Parameters
    ----------
    argon2_params : Optional[Dict[str, int]], default=None
        Custom Argon2 parameters. If None, uses default values.

    Returns
    -------
    BiometricZkCircuit
        Configured biometric ZK circuit.

    Examples
    --------
    >>> circuit = create_biometric_circuit()
    >>> spec = circuit.build_circuit()
    >>> summary = circuit.generate_circuit_summary()
    """
    if argon2_params is None:
        argon2_params = {
            "time_cost": ARGON2_TIME_COST,
            "memory_cost": ARGON2_MEMORY_COST,
        }

    return BiometricZkCircuit(
        argon2_time_cost=argon2_params.get("time_cost", ARGON2_TIME_COST),
        argon2_memory_cost=argon2_params.get("memory_cost", ARGON2_MEMORY_COST),
    )


def validate_circuit_inputs(
    fused_vector: np.ndarray, salt: bytes, template_hash: str
) -> bool:
    """
    Validate inputs for ZK circuit execution.

    Parameters
    ----------
    fused_vector : np.ndarray
        Fused biometric feature vector.
    salt : bytes
        Cryptographic salt.
    template_hash : str
        Expected template hash.

    Returns
    -------
    bool
        True if all inputs are valid.

    Raises
    ------
    ProofGenerationError
        If any inputs are invalid.
    """
    errors = []

    # Validate fused vector
    if not isinstance(fused_vector, np.ndarray):
        errors.append(f"fused_vector must be numpy array, got {type(fused_vector)}")
    elif fused_vector.size == 0:
        errors.append("fused_vector cannot be empty")
    elif not np.isfinite(fused_vector).all():
        errors.append("fused_vector contains non-finite values")

    # Validate salt
    if not isinstance(salt, bytes):
        errors.append(f"salt must be bytes, got {type(salt)}")
    elif len(salt) == 0:
        errors.append("salt cannot be empty")

    # Validate template hash
    if not isinstance(template_hash, str):
        errors.append(f"template_hash must be string, got {type(template_hash)}")
    elif len(template_hash) == 0:
        errors.append("template_hash cannot be empty")
    elif len(template_hash) % 2 != 0:
        errors.append("template_hash must have even length (hex string)")
    else:
        try:
            bytes.fromhex(template_hash)
        except ValueError:
            errors.append("template_hash must be valid hex string")

    if errors:
        raise ProofGenerationError(
            f"Invalid circuit inputs: {'; '.join(errors)}",
            circuit_type="biometric_verification",
        )

    return True
