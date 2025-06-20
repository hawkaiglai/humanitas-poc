"""
Custom exception classes for the HUMANITAS POC system.

This module defines a hierarchy of custom exceptions to enable precise
error handling and debugging throughout the application. Each exception
includes detailed context information for academic reproducibility.
"""

from typing import Optional, Dict, Any


class HumanitasPocError(Exception):
    """
    Base exception class for all HUMANITAS POC related errors.

    This serves as the parent class for all custom exceptions in the system,
    providing common functionality for error reporting and debugging.

    Parameters
    ----------
    message : str
        Human-readable error message.
    context : dict, optional
        Additional context information about the error.
    error_code : str, optional
        Unique error code for programmatic handling.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a formatted string representation of the error."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"[Error Code: {self.error_code}]")

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[Context: {context_str}]")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary for structured logging.

        Returns
        -------
        dict
            Dictionary representation of the exception.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class DatasetError(HumanitasPocError):
    """
    Exception raised for errors related to loading or parsing datasets.

    This includes issues with file paths, data format, missing files,
    or corrupted dataset structures.
    """

    def __init__(
        self,
        message: str,
        dataset_path: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.get("context", {})
        if dataset_path:
            context["dataset_path"] = dataset_path
        if file_path:
            context["file_path"] = file_path

        super().__init__(message, context, kwargs.get("error_code"))


class DatasetNotFoundError(DatasetError):
    """Exception raised when a required dataset is not found."""

    def __init__(self, dataset_path: str, dataset_type: str = "unknown") -> None:
        message = f"Dataset not found: {dataset_path}"
        context = {"dataset_path": dataset_path, "dataset_type": dataset_type}
        super().__init__(message, context=context, error_code="DATASET_001")


class DatasetCorruptedError(DatasetError):
    """Exception raised when a dataset appears to be corrupted or invalid."""

    def __init__(self, dataset_path: str, reason: str) -> None:
        message = f"Dataset appears corrupted: {reason}"
        context = {"dataset_path": dataset_path, "corruption_reason": reason}
        super().__init__(message, context=context, error_code="DATASET_002")


class BiometricProcessingError(HumanitasPocError):
    """
    Exception raised for errors during biometric processing.

    This includes feature extraction, normalization, fusion, and other
    biometric-specific operations.
    """

    def __init__(
        self,
        message: str,
        processing_stage: Optional[str] = None,
        sample_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.get("context", {})
        if processing_stage:
            context["processing_stage"] = processing_stage
        if sample_id:
            context["sample_id"] = sample_id

        super().__init__(message, context, kwargs.get("error_code"))


class FeatureExtractionError(BiometricProcessingError):
    """Exception raised during feature extraction from biometric samples."""

    def __init__(
        self, message: str, image_path: str, extraction_method: str, **kwargs
    ) -> None:
        context = {"image_path": image_path, "extraction_method": extraction_method}
        super().__init__(
            message,
            processing_stage="feature_extraction",
            context=context,
            error_code="BIOMETRIC_001",
        )


class NormalizationError(BiometricProcessingError):
    """Exception raised during feature vector normalization."""

    def __init__(
        self, message: str, vector_shape: tuple, target_dimension: int, **kwargs
    ) -> None:
        context = {"vector_shape": vector_shape, "target_dimension": target_dimension}
        super().__init__(
            message,
            processing_stage="normalization",
            context=context,
            error_code="BIOMETRIC_002",
        )


class FusionError(BiometricProcessingError):
    """Exception raised during multimodal feature fusion."""

    def __init__(
        self, message: str, modalities: list, fusion_method: str = "unknown", **kwargs
    ) -> None:
        context = {"modalities": modalities, "fusion_method": fusion_method}
        super().__init__(
            message,
            processing_stage="fusion",
            context=context,
            error_code="BIOMETRIC_003",
        )


class QualityCheckError(BiometricProcessingError):
    """
    Exception raised when a biometric sample does not meet quality standards.

    This is typically not a system error but indicates that the input
    biometric sample is of insufficient quality for reliable processing.
    """

    def __init__(
        self,
        message: str,
        quality_score: float,
        minimum_threshold: float,
        modality: str,
        **kwargs,
    ) -> None:
        context = {
            "quality_score": quality_score,
            "minimum_threshold": minimum_threshold,
            "modality": modality,
        }
        super().__init__(
            message,
            processing_stage="quality_assessment",
            context=context,
            error_code="BIOMETRIC_004",
        )


class CryptographyError(HumanitasPocError):
    """
    Exception raised for errors in cryptographic operations.

    This includes template generation, hashing, and ZK-proof operations.
    """

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs) -> None:
        context = kwargs.get("context", {})
        if operation:
            context["cryptographic_operation"] = operation

        super().__init__(message, context, kwargs.get("error_code"))


class TemplateGenerationError(CryptographyError):
    """Exception raised during secure template generation."""

    def __init__(self, message: str, algorithm: str = "argon2", **kwargs) -> None:
        context = {"hashing_algorithm": algorithm}
        super().__init__(
            message,
            operation="template_generation",
            context=context,
            error_code="CRYPTO_001",
        )


class ProofGenerationError(CryptographyError):
    """Exception raised during ZK-proof generation."""

    def __init__(
        self,
        message: str,
        circuit_type: str = "unknown",
        proof_system: str = "groth16",
        **kwargs,
    ) -> None:
        context = {"circuit_type": circuit_type, "proof_system": proof_system}
        super().__init__(
            message,
            operation="proof_generation",
            context=context,
            error_code="CRYPTO_002",
        )


class ProofVerificationError(CryptographyError):
    """Exception raised during ZK-proof verification."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(
            message, operation="proof_verification", error_code="CRYPTO_003"
        )


class TestExecutionError(HumanitasPocError):
    """
    Exception raised during test execution.

    This includes errors in the test harness, benchmark execution,
    or test result processing.
    """

    def __init__(
        self,
        message: str,
        test_type: Optional[str] = None,
        test_stage: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.get("context", {})
        if test_type:
            context["test_type"] = test_type
        if test_stage:
            context["test_stage"] = test_stage

        super().__init__(message, context, kwargs.get("error_code"))


class TestConfigurationError(TestExecutionError):
    """Exception raised for test configuration errors."""

    def __init__(self, message: str, parameter: str, **kwargs) -> None:
        context = {"invalid_parameter": parameter}
        super().__init__(
            message, test_stage="configuration", context=context, error_code="TEST_001"
        )


class TestDataError(TestExecutionError):
    """Exception raised for errors related to test data."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, test_stage="data_preparation", error_code="TEST_002")


class PerformanceBenchmarkError(TestExecutionError):
    """Exception raised during performance benchmarking."""

    def __init__(self, message: str, benchmark_type: str, **kwargs) -> None:
        context = {"benchmark_type": benchmark_type}
        super().__init__(
            message, test_type="performance", context=context, error_code="TEST_003"
        )


class ConfigurationError(HumanitasPocError):
    """
    Exception raised for configuration-related errors.

    This includes invalid configuration values, missing required
    environment variables, or configuration conflicts.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value:
            context["config_value"] = config_value

        super().__init__(message, context, kwargs.get("error_code", "CONFIG_001"))


class DependencyError(HumanitasPocError):
    """
    Exception raised for missing or incompatible dependencies.

    This includes missing Python packages, incompatible versions,
    or system-level dependencies.
    """

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.get("context", {})
        if dependency_name:
            context["dependency_name"] = dependency_name
        if required_version:
            context["required_version"] = required_version

        super().__init__(message, context, kwargs.get("error_code", "DEP_001"))
