"""
Data models for the HUMANITAS POC system.

This module defines the core data structures used throughout the biometric
processing pipeline. All models use dataclasses for clean, type-safe
data representation with automatic generation of common methods.

The models are designed to support the academic research requirements
with comprehensive metadata tracking and serialization capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import numpy as np
from pathlib import Path


@dataclass
class BiometricSample:
    """
    Represents a complete biometric sample for one person.

    This class encapsulates all biometric data and derived features for a single
    individual, including multiple fingerprint samples, facial image, extracted
    features, and cryptographic templates.

    Parameters
    ----------
    person_id : str
        Unique identifier for the person (e.g., "person_001").
    fingerprint_paths : List[str]
        List of file paths to fingerprint images for this person.
        Expected to contain exactly 10 paths per NIST SD-302 standard.
    face_path : str
        File path to the facial image for this person.
    fingerprint_features : Optional[np.ndarray], default=None
        Extracted and normalized fingerprint features.
        Shape should be (n_fingerprints, feature_dim) after processing.
    face_features : Optional[np.ndarray], default=None
        Extracted facial features vector.
        Shape should be (feature_dim,) after processing.
    fused_template : Optional[bytes], default=None
        Cryptographically secure template generated from fused features.
        This is the core output used for ZK-proof generation.
    template_hash : Optional[str], default=None
        Hexadecimal string representation of the Argon2 hash.
        Used as public input for ZK-proofs.
    salt : Optional[bytes], default=None
        Cryptographic salt used in template generation.
        Required for template verification and proof generation.
    quality_scores : Dict[str, float], default_factory=dict
        Quality assessment scores for each biometric modality.
        Keys: 'face', 'fingerprint_avg', 'fingerprint_individual'.
    processing_metadata : Dict[str, Any], default_factory=dict
        Metadata about the processing pipeline applied to this sample.
        Includes extraction methods, normalization parameters, etc.
    created_at : datetime, default_factory=datetime.now
        Timestamp when this sample object was created.

    Examples
    --------
    >>> sample = BiometricSample(
    ...     person_id="person_001",
    ...     fingerprint_paths=["/path/to/fp1.png", "/path/to/fp2.png"],
    ...     face_path="/path/to/face.jpg"
    ... )
    >>> sample.person_id
    'person_001'
    """

    person_id: str
    fingerprint_paths: List[str]
    face_path: str
    fingerprint_features: Optional[List[np.ndarray]] = None  # <-- THE FIX
    face_features: Optional[np.ndarray] = None
    fused_template: Optional[bytes] = None
    template_hash: Optional[str] = None
    salt: Optional[bytes] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """
        Validate the biometric sample data after initialization.

        Raises
        ------
        ValueError
            If the sample data is invalid or inconsistent.
        """
        # Validate person_id
        if not self.person_id or not isinstance(self.person_id, str):
            raise ValueError("person_id must be a non-empty string")

        # Validate fingerprint paths
        if not self.fingerprint_paths:
            raise ValueError("fingerprint_paths cannot be empty")

        if len(self.fingerprint_paths) > 10:
            raise ValueError("Too many fingerprint paths (max 10 per NIST standard)")

        # Validate that all paths are strings
        for i, path in enumerate(self.fingerprint_paths):
            if not isinstance(path, str):
                raise ValueError(f"fingerprint_paths[{i}] must be a string")

        # Validate face path
        if not self.face_path or not isinstance(self.face_path, str):
            raise ValueError("face_path must be a non-empty string")

        # Validate feature arrays if present
        if self.fingerprint_features is not None:
            if not isinstance(self.fingerprint_features, np.ndarray):
                raise ValueError("fingerprint_features must be a numpy array")

        if self.face_features is not None:
            if not isinstance(self.face_features, np.ndarray):
                raise ValueError("face_features must be a numpy array")
            if self.face_features.ndim != 1:
                raise ValueError("face_features must be a 1D array")

    @property
    def has_features(self) -> bool:
        """
        Check if biometric features have been extracted.

        Returns
        -------
        bool
            True if both fingerprint and face features are available.
        """
        return self.fingerprint_features is not None and self.face_features is not None

    @property
    def has_template(self) -> bool:
        """
        Check if a cryptographic template has been generated.

        Returns
        -------
        bool
            True if template, hash, and salt are all available.
        """
        return (
            self.fused_template is not None
            and self.template_hash is not None
            and self.salt is not None
        )

    @property
    def fingerprint_count(self) -> int:
        """
        Get the number of fingerprint samples.

        Returns
        -------
        int
            Number of fingerprint image paths.
        """
        return len(self.fingerprint_paths)

    def get_file_paths(self) -> List[Path]:
        """
        Get all file paths associated with this sample as Path objects.

        Returns
        -------
        List[Path]
            List containing all fingerprint and face image paths.
        """
        paths = [Path(fp) for fp in self.fingerprint_paths]
        paths.append(Path(self.face_path))
        return paths

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sample to a dictionary for serialization.

        Note: NumPy arrays and bytes objects are converted to lists/strings
        for JSON compatibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the sample.
        """
        result = {
            "person_id": self.person_id,
            "fingerprint_paths": self.fingerprint_paths,
            "face_path": self.face_path,
            "quality_scores": self.quality_scores,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat(),
        }

        # Handle numpy arrays
        if self.fingerprint_features is not None:
            result["fingerprint_features_shape"] = self.fingerprint_features.shape
            result["fingerprint_features"] = self.fingerprint_features.tolist()

        if self.face_features is not None:
            result["face_features_shape"] = self.face_features.shape
            result["face_features"] = self.face_features.tolist()

        # Handle bytes objects
        if self.fused_template is not None:
            result["fused_template"] = self.fused_template.hex()

        if self.salt is not None:
            result["salt"] = self.salt.hex()

        if self.template_hash is not None:
            result["template_hash"] = self.template_hash

        return result


@dataclass
class TestResult:
    """
    Represents the result of a single biometric test or comparison.

    This class captures all relevant information about a biometric test,
    including the test parameters, outcomes, performance metrics, and
    cryptographic proof validation results.

    Parameters
    ----------
    test_id : str
        Unique identifier for this test execution.
    test_type : str
        Type of test performed (e.g., 'fmr', 'fnmr', 'performance').
    person1_id : str
        Identifier of the first person involved in the test.
    person2_id : Optional[str], default=None
        Identifier of the second person (for comparison tests).
        None for single-person tests like enrollment.
    outcome : str
        Actual outcome of the test (e.g., 'match', 'no_match', 'error').
    expected_outcome : str
        Expected outcome based on ground truth.
    confidence_score : float
        Confidence score or similarity measure (0.0 to 1.0).
    processing_times : Dict[str, int]
        Processing time in milliseconds for each pipeline stage.
        Keys: 'feature_extraction', 'fusion', 'template_generation', 'zk_proof'.
    template_hash : str
        Hash of the biometric template used in this test.
    zk_proof_valid : bool
        Whether the zero-knowledge proof was successfully generated and verified.
    error_details : Optional[Dict[str, Any]], default=None
        Additional error information if the test failed.
    test_parameters : Dict[str, Any], default_factory=dict
        Parameters used for this specific test execution.
    timestamp : datetime, default_factory=datetime.now
        Timestamp when the test was executed.

    Examples
    --------
    >>> result = TestResult(
    ...     test_id="test_001",
    ...     test_type="fmr",
    ...     person1_id="person_001",
    ...     person2_id="person_002",
    ...     outcome="no_match",
    ...     expected_outcome="no_match",
    ...     confidence_score=0.15,
    ...     processing_times={"total": 1500},
    ...     template_hash="abc123...",
    ...     zk_proof_valid=True
    ... )
    """

    test_id: str
    test_type: str
    person1_id: str
    person2_id: Optional[str]
    outcome: str
    expected_outcome: str
    confidence_score: float
    processing_times: Dict[str, int]
    template_hash: str
    zk_proof_valid: bool
    error_details: Optional[Dict[str, Any]] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """
        Validate the test result data after initialization.

        Raises
        ------
        ValueError
            If the test result data is invalid.
        """
        # Validate required string fields
        required_strings = [
            ("test_id", self.test_id),
            ("test_type", self.test_type),
            ("person1_id", self.person1_id),
            ("outcome", self.outcome),
            ("expected_outcome", self.expected_outcome),
            ("template_hash", self.template_hash),
        ]

        for field_name, value in required_strings:
            if not value or not isinstance(value, str):
                raise ValueError(f"{field_name} must be a non-empty string")

        # Validate confidence score
        if not isinstance(self.confidence_score, (int, float)):
            raise ValueError("confidence_score must be a number")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")

        # Validate processing times
        if not isinstance(self.processing_times, dict):
            raise ValueError("processing_times must be a dictionary")

        for stage, time_ms in self.processing_times.items():
            if not isinstance(time_ms, int) or time_ms < 0:
                raise ValueError(
                    f"processing_times['{stage}'] must be a non-negative integer"
                )

        # Validate boolean fields
        if not isinstance(self.zk_proof_valid, bool):
            raise ValueError("zk_proof_valid must be a boolean")

    @property
    def is_correct(self) -> bool:
        """
        Check if the test outcome matches the expected outcome.

        Returns
        -------
        bool
            True if the actual outcome matches the expected outcome.
        """
        return self.outcome == self.expected_outcome

    @property
    def total_processing_time(self) -> int:
        """
        Get the total processing time across all stages.

        Returns
        -------
        int
            Total processing time in milliseconds.
        """
        return sum(self.processing_times.values())

    @property
    def is_comparison_test(self) -> bool:
        """
        Check if this is a comparison test between two people.

        Returns
        -------
        bool
            True if person2_id is not None.
        """
        return self.person2_id is not None

    @property
    def has_error(self) -> bool:
        """
        Check if the test encountered an error.

        Returns
        -------
        bool
            True if error_details is not None and not empty.
        """
        return self.error_details is not None and bool(self.error_details)

    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic accuracy metrics for this test result.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing accuracy metrics.
        """
        return {
            "correct": self.is_correct,
            "confidence": self.confidence_score,
            "outcome": self.outcome,
            "expected": self.expected_outcome,
            "error_rate": 0.0 if self.is_correct else 1.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test result to a dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the test result.
        """
        result = {
            "test_id": self.test_id,
            "test_type": self.test_type,
            "person1_id": self.person1_id,
            "person2_id": self.person2_id,
            "outcome": self.outcome,
            "expected_outcome": self.expected_outcome,
            "confidence_score": self.confidence_score,
            "processing_times": self.processing_times,
            "template_hash": self.template_hash,
            "zk_proof_valid": self.zk_proof_valid,
            "error_details": self.error_details,
            "test_parameters": self.test_parameters,
            "timestamp": self.timestamp.isoformat(),
            # Computed properties for convenience
            "is_correct": self.is_correct,
            "total_processing_time": self.total_processing_time,
            "is_comparison_test": self.is_comparison_test,
            "has_error": self.has_error,
        }

        return result


@dataclass
class DatasetStatistics:
    """
    Statistics about a loaded biometric dataset.

    This class provides comprehensive statistics about the dataset
    for academic reporting and validation purposes.

    Parameters
    ----------
    total_samples : int
        Total number of biometric samples loaded.
    total_people : int
        Total number of unique individuals in the dataset.
    fingerprint_images : int
        Total number of fingerprint images.
    face_images : int
        Total number of face images.
    avg_fingerprints_per_person : float
        Average number of fingerprints per person.
    dataset_paths : Dict[str, str]
        Paths to the source datasets.
    loading_time_seconds : float
        Time taken to load the dataset.
    quality_distribution : Dict[str, Dict[str, float]], default_factory=dict
        Distribution of quality scores by modality.
    """

    total_samples: int
    total_people: int
    fingerprint_images: int
    face_images: int
    avg_fingerprints_per_person: float
    dataset_paths: Dict[str, str]
    loading_time_seconds: float
    quality_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            "total_samples": self.total_samples,
            "total_people": self.total_people,
            "fingerprint_images": self.fingerprint_images,
            "face_images": self.face_images,
            "avg_fingerprints_per_person": self.avg_fingerprints_per_person,
            "dataset_paths": self.dataset_paths,
            "loading_time_seconds": self.loading_time_seconds,
            "quality_distribution": self.quality_distribution,
        }
