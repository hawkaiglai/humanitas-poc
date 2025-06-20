"""
Multimodal biometric feature fusion for the HUMANITAS POC system.

This module implements the core intellectual property of the system: sophisticated
fusion of fingerprint and facial biometric features into a single, secure template.
The fusion process combines multiple fingerprint samples with facial features using
advanced hashing and weighting techniques.

The fusion algorithm is designed to be non-reversible while preserving uniqueness,
making it suitable for zero-knowledge proof systems and privacy-preserving
biometric authentication.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction import FeatureHasher
import structlog

from .constants import (
    FP_FEATURE_DIM,
    FACE_FEATURE_DIM,
    FUSED_FEATURE_DIM,
    DEFAULT_FUSION_WEIGHTS,
)
from .exceptions import FusionError
from .normalization import normalize_vector

# Initialize structured logger
logger = structlog.get_logger(__name__)


class BiometricFusionEngine:
    """
    Advanced multimodal biometric fusion engine.

    This class implements sophisticated fusion algorithms that combine multiple
    biometric modalities while preserving uniqueness and enhancing security.
    The fusion process is designed to be irreversible while maintaining
    discriminative power for authentication.

    Parameters
    ----------
    fusion_weights : Dict[str, float], default=DEFAULT_FUSION_WEIGHTS
        Weights for combining different biometric modalities.
    hash_features : int, default=FUSED_FEATURE_DIM
        Number of features for the final hashed representation.
    fusion_method : str, default="weighted_hash"
        Fusion algorithm to use. Options: "weighted_hash", "concatenate", "pca".
    random_state : Optional[int], default=42
        Random seed for reproducible fusion results.

    Examples
    --------
    >>> fusion_engine = BiometricFusionEngine()
    >>> fp_features = [np.random.randn(512) for _ in range(10)]
    >>> face_features = np.random.randn(128)
    >>> fused = fusion_engine.fuse_features(fp_features, face_features)
    >>> print(f"Fused template shape: {fused.shape}")
    """

    def __init__(
        self,
        fusion_weights: Dict[str, float] = None,
        hash_features: int = FUSED_FEATURE_DIM,
        fusion_method: str = "weighted_hash",
        random_state: Optional[int] = 42,
    ) -> None:
        self.fusion_weights = fusion_weights or DEFAULT_FUSION_WEIGHTS.copy()
        self.hash_features = hash_features
        self.fusion_method = fusion_method
        self.random_state = random_state

        # Validate fusion weights
        self._validate_fusion_weights()

        # Validate fusion method
        valid_methods = ["weighted_hash", "concatenate", "pca"]
        if fusion_method not in valid_methods:
            raise FusionError(
                f"Invalid fusion method '{fusion_method}'. Must be one of {valid_methods}",
                modalities=list(self.fusion_weights.keys()),
                fusion_method=fusion_method,
            )

        # Initialize feature hasher for non-linear projection
        self.feature_hasher = FeatureHasher(
            n_features=hash_features, input_type="string", alternate_sign=False
        )

        logger.info(
            "BiometricFusionEngine initialized",
            fusion_weights=self.fusion_weights,
            hash_features=hash_features,
            fusion_method=fusion_method,
            random_state=random_state,
        )

    def _validate_fusion_weights(self) -> None:
        """
        Validate fusion weights for consistency and normalization.

        Raises
        ------
        FusionError
            If fusion weights are invalid.
        """
        required_modalities = ["fingerprint", "face"]

        for modality in required_modalities:
            if modality not in self.fusion_weights:
                raise FusionError(
                    f"Missing fusion weight for modality '{modality}'",
                    modalities=list(self.fusion_weights.keys()),
                    fusion_method=self.fusion_method,
                )

        # Check that weights are positive
        for modality, weight in self.fusion_weights.items():
            if weight < 0:
                raise FusionError(
                    f"Fusion weight for '{modality}' must be non-negative, got {weight}",
                    modalities=list(self.fusion_weights.keys()),
                    fusion_method=self.fusion_method,
                )

        # Normalize weights to sum to 1.0
        total_weight = sum(self.fusion_weights.values())
        if total_weight == 0:
            raise FusionError(
                "Fusion weights cannot all be zero",
                modalities=list(self.fusion_weights.keys()),
                fusion_method=self.fusion_method,
            )

        # Normalize weights
        for modality in self.fusion_weights:
            self.fusion_weights[modality] /= total_weight

        logger.debug(
            "Fusion weights normalized", normalized_weights=self.fusion_weights
        )

    def _average_fingerprint_features(
        self, fp_features_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Average multiple fingerprint feature vectors into a single representation.
        This version robustly handles feature vectors of different lengths by
        padding them to a consistent size before averaging.

        Parameters
        ----------
        fp_features_list : List[np.ndarray]
            List of fingerprint feature vectors.

        Returns
        -------
        np.ndarray
            Averaged fingerprint feature vector.

        Raises
        ------
        FusionError
            If fingerprint features are inconsistent or invalid.
        """
        if not fp_features_list:
            raise FusionError(
                "Cannot average empty fingerprint feature list",
                modalities=["fingerprint"],
                fusion_method=self.fusion_method,
            )

        # --- THE FIX IS HERE ---
        # Find the length of the longest feature vector in the list.
        try:
            max_len = max(len(v) for v in fp_features_list)
        except ValueError:
            raise FusionError(
                "Could not determine max length from feature list.",
                modalities=["fingerprint"],
                fusion_method=self.fusion_method,
            )

        # Create a new list of padded vectors.
        # Each vector is padded with zeros to match the max length.
        padded_vectors = [
            np.pad(v, (0, max_len - len(v)), "constant") for v in fp_features_list
        ]

        # Stack and average the now consistently-sized vectors.
        feature_matrix = np.vstack(padded_vectors)
        averaged_features = np.mean(feature_matrix, axis=0)

        logger.debug(
            "Fingerprint features padded and averaged",
            n_fingerprints=len(fp_features_list),
            target_dim=max_len,
            averaged_shape=averaged_features.shape,
            averaged_stats={
                "mean": float(np.mean(averaged_features)),
                "std": float(np.std(averaged_features)),
            },
        )

        return averaged_features

    def _create_feature_hash(self, features: np.ndarray, modality: str) -> np.ndarray:
        """
        Create a non-linear hash projection of feature vector.

        Parameters
        ----------
        features : np.ndarray
            Input feature vector.
        modality : str
            Biometric modality name (for logging).

        Returns
        -------
        np.ndarray
            Hashed feature representation.
        """
        # Convert features to string representation for hashing
        # This creates a non-linear, irreversible transformation
        feature_strings = [
            f"{modality}_{i}_{val:.6f}" for i, val in enumerate(features)
        ]

        # Apply feature hashing
        hashed_features = self.feature_hasher.transform([feature_strings]).toarray()[0]

        logger.debug(
            "Feature hash created",
            modality=modality,
            input_dim=len(features),
            output_dim=len(hashed_features),
            hash_density=float(np.count_nonzero(hashed_features))
            / len(hashed_features),
        )

        return hashed_features

    def _weighted_hash_fusion(
        self, fp_features: np.ndarray, face_features: np.ndarray
    ) -> np.ndarray:
        """
        Perform weighted hash-based fusion of biometric features.

        This is the core fusion algorithm that creates irreversible but
        discriminative biometric templates.

        Parameters
        ----------
        fp_features : np.ndarray
            Normalized fingerprint features.
        face_features : np.ndarray
            Normalized face features.

        Returns
        -------
        np.ndarray
            Fused biometric template.
        """
        logger.info("Performing weighted hash fusion")

        # Create non-linear hash projections
        fp_hash = self._create_feature_hash(fp_features, "fingerprint")
        face_hash = self._create_feature_hash(face_features, "face")

        # Apply fusion weights
        fp_weight = self.fusion_weights["fingerprint"]
        face_weight = self.fusion_weights["face"]

        # Weighted combination
        fused_features = (fp_weight * fp_hash) + (face_weight * face_hash)

        # Additional non-linear transformation for security
        # Apply element-wise transformation to enhance irreversibility
        fused_features = np.tanh(fused_features) * np.sqrt(np.abs(fused_features))

        logger.info(
            "Weighted hash fusion completed",
            fp_weight=fp_weight,
            face_weight=face_weight,
            fused_stats={
                "mean": float(np.mean(fused_features)),
                "std": float(np.std(fused_features)),
                "sparsity": float(np.count_nonzero(fused_features))
                / len(fused_features),
            },
        )

        return fused_features

    def _concatenate_fusion(
        self, fp_features: np.ndarray, face_features: np.ndarray
    ) -> np.ndarray:
        """
        Simple concatenation-based fusion.

        Parameters
        ----------
        fp_features : np.ndarray
            Normalized fingerprint features.
        face_features : np.ndarray
            Normalized face features.

        Returns
        -------
        np.ndarray
            Concatenated and potentially reduced feature vector.
        """
        logger.debug("Performing concatenation fusion")

        # Apply weights by scaling features
        fp_weight = self.fusion_weights["fingerprint"]
        face_weight = self.fusion_weights["face"]

        weighted_fp = fp_features * fp_weight
        weighted_face = face_features * face_weight

        # Concatenate features
        concatenated = np.concatenate([weighted_fp, weighted_face])

        # Reduce to target dimension if necessary
        if len(concatenated) > self.hash_features:
            # Use simple truncation or could use PCA
            fused_features = concatenated[: self.hash_features]
        elif len(concatenated) < self.hash_features:
            # Zero-pad if needed
            fused_features = np.zeros(self.hash_features)
            fused_features[: len(concatenated)] = concatenated
        else:
            fused_features = concatenated

        return fused_features

    def fuse_features(
        self, fp_features_list: List[np.ndarray], face_features: np.ndarray
    ) -> np.ndarray:
        """
        Fuse multiple fingerprint features with face features into a single template.

        This is the main fusion method that implements the core algorithm for
        combining multimodal biometric features. The process involves:
        1. Averaging multiple fingerprint samples FIRST
        2. Normalizing the single averaged vector and face features
        3. Applying non-linear hash-based fusion
        4. Creating an irreversible but discriminative template

        Parameters
        ----------
        fp_features_list : List[np.ndarray]
            List of fingerprint feature vectors (typically 10 per person).
        face_features : np.ndarray
            Single face feature vector.

        Returns
        -------
        np.ndarray
            Fused biometric template with shape (hash_features,).

        Raises
        ------
        FusionError
            If fusion process fails for any reason.

        Examples
        --------
        >>> fusion_engine = BiometricFusionEngine()
        >>> fp_features = [np.random.randn(512) for _ in range(10)]
        >>> face_features = np.random.randn(128)
        >>> template = fusion_engine.fuse_features(fp_features, face_features)
        >>> assert template.shape == (1024,)
        """
        logger.info(
            "Starting multimodal feature fusion",
            n_fingerprints=len(fp_features_list),
            face_features_shape=face_features.shape,
            fusion_method=self.fusion_method,
        )

        # Validate inputs
        if not fp_features_list:
            raise FusionError(
                "Fingerprint features list cannot be empty",
                modalities=["fingerprint", "face"],
                fusion_method=self.fusion_method,
            )

        if not isinstance(face_features, np.ndarray) or face_features.ndim != 1:
            raise FusionError(
                f"Face features must be 1D numpy array, got {type(face_features)} with shape {getattr(face_features, 'shape', 'unknown')}",
                modalities=["fingerprint", "face"],
                fusion_method=self.fusion_method,
            )

        try:
            # Step 1: Average fingerprint features BEFORE normalization
            # This avoids the PCA learning problem by reducing 10 vectors to 1
            averaged_fp_features = self._average_fingerprint_features(fp_features_list)

            # Step 2: Normalize the single averaged fingerprint vector
            normalized_fp_features = normalize_vector(
                averaged_fp_features, FP_FEATURE_DIM
            )

            # Step 3: Normalize face features using simple normalization
            normalized_face_features = normalize_vector(face_features, FACE_FEATURE_DIM)

            # Step 4: Apply fusion algorithm
            if self.fusion_method == "weighted_hash":
                fused_template = self._weighted_hash_fusion(
                    normalized_fp_features, normalized_face_features
                )
            elif self.fusion_method == "concatenate":
                fused_template = self._concatenate_fusion(
                    normalized_fp_features, normalized_face_features
                )
            else:
                raise FusionError(
                    f"Fusion method '{self.fusion_method}' not implemented",
                    modalities=["fingerprint", "face"],
                    fusion_method=self.fusion_method,
                )

            # Step 5: Final validation and type conversion
            fused_template = fused_template.astype(np.float32)

            if not np.isfinite(fused_template).all():
                raise FusionError(
                    "Fused template contains non-finite values",
                    modalities=["fingerprint", "face"],
                    fusion_method=self.fusion_method,
                )

            logger.info(
                "Multimodal fusion completed successfully",
                template_shape=fused_template.shape,
                template_stats={
                    "mean": float(np.mean(fused_template)),
                    "std": float(np.std(fused_template)),
                    "min": float(np.min(fused_template)),
                    "max": float(np.max(fused_template)),
                },
            )

            return fused_template

        except Exception as e:
            if isinstance(e, FusionError):
                raise
            else:
                raise FusionError(
                    f"Unexpected error during fusion: {str(e)}",
                    modalities=["fingerprint", "face"],
                    fusion_method=self.fusion_method,
                )


def fuse_features(
    fp_features_list: List[np.ndarray],
    face_features: np.ndarray,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Convenience function for multimodal biometric feature fusion.

    This function provides a simple interface to the fusion engine with
    default parameters for standard use cases.

    Parameters
    ----------
    fp_features_list : List[np.ndarray]
        List of fingerprint feature vectors.
    face_features : np.ndarray
        Face feature vector.
    fusion_weights : Optional[Dict[str, float]], default=None
        Custom fusion weights. If None, uses default weights.

    Returns
    -------
    np.ndarray
        Fused biometric template.

    Raises
    ------
    FusionError
        If fusion fails.

    Examples
    --------
    >>> fp_features = [np.random.randn(512) for _ in range(10)]
    >>> face_features = np.random.randn(128)
    >>> template = fuse_features(fp_features, face_features)
    >>> print(f"Template shape: {template.shape}")
    """
    fusion_engine = BiometricFusionEngine(fusion_weights=fusion_weights)
    return fusion_engine.fuse_features(fp_features_list, face_features)


def calculate_fusion_quality_metrics(fused_template: np.ndarray) -> Dict[str, float]:
    """
    Calculate quality metrics for a fused biometric template.

    These metrics help assess the quality and security properties
    of the fused template for academic analysis.

    Parameters
    ----------
    fused_template : np.ndarray
        Fused biometric template.

    Returns
    -------
    Dict[str, float]
        Dictionary containing various quality metrics.

    Examples
    --------
    >>> template = fuse_features(fp_features, face_features)
    >>> metrics = calculate_fusion_quality_metrics(template)
    >>> print(f"Template entropy: {metrics['entropy']}")
    """
    metrics = {}

    # Basic statistical measures
    metrics["mean"] = float(np.mean(fused_template))
    metrics["std"] = float(np.std(fused_template))
    metrics["min"] = float(np.min(fused_template))
    metrics["max"] = float(np.max(fused_template))

    # Sparsity (proportion of non-zero elements)
    metrics["sparsity"] = float(np.count_nonzero(fused_template)) / len(fused_template)

    # Dynamic range
    metrics["dynamic_range"] = metrics["max"] - metrics["min"]

    # Entropy estimation (using histogram-based approach)
    hist, _ = np.histogram(fused_template, bins=50)
    hist = hist[hist > 0]  # Remove zero bins
    probabilities = hist / np.sum(hist)
    metrics["entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))

    # Template uniqueness indicators
    metrics["unique_values"] = len(np.unique(fused_template))
    metrics["uniqueness_ratio"] = metrics["unique_values"] / len(fused_template)

    # Statistical independence measures
    # Autocorrelation at lag 1
    if len(fused_template) > 1:
        autocorr = np.corrcoef(fused_template[:-1], fused_template[1:])[0, 1]
        metrics["autocorrelation_lag1"] = (
            float(autocorr) if np.isfinite(autocorr) else 0.0
        )
    else:
        metrics["autocorrelation_lag1"] = 0.0

    return metrics


def test_fusion_reproducibility(
    fp_features_list: List[np.ndarray],
    face_features: np.ndarray,
    n_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Test the reproducibility of the fusion process.

    This function verifies that the fusion algorithm produces
    consistent results across multiple runs with the same input.

    Parameters
    ----------
    fp_features_list : List[np.ndarray]
        Fingerprint feature vectors.
    face_features : np.ndarray
        Face feature vector.
    n_iterations : int, default=10
        Number of iterations to test.

    Returns
    -------
    Dict[str, Any]
        Reproducibility test results.

    Examples
    --------
    >>> results = test_fusion_reproducibility(fp_features, face_features)
    >>> print(f"Reproducible: {results['is_reproducible']}")
    """
    logger.info(f"Testing fusion reproducibility over {n_iterations} iterations")

    templates = []

    for i in range(n_iterations):
        template = fuse_features(fp_features_list, face_features)
        templates.append(template)

    # Check if all templates are identical
    first_template = templates[0]
    is_reproducible = all(np.array_equal(first_template, t) for t in templates[1:])

    # Calculate variation statistics
    if not is_reproducible:
        template_matrix = np.vstack(templates)
        max_variation = float(np.max(np.std(template_matrix, axis=0)))
        mean_variation = float(np.mean(np.std(template_matrix, axis=0)))
    else:
        max_variation = 0.0
        mean_variation = 0.0

    results = {
        "is_reproducible": is_reproducible,
        "n_iterations": n_iterations,
        "max_variation": max_variation,
        "mean_variation": mean_variation,
        "template_shape": first_template.shape,
    }

    logger.info("Fusion reproducibility test completed", **results)

    return results
