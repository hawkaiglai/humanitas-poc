"""
Feature vector normalization for the HUMANITAS POC system.

This module provides simple normalization techniques to standardize biometric
feature vectors to consistent dimensions. The normalization uses direct
padding/truncation methods that don't require learning from multiple samples,
avoiding the PCA sample size limitations.
"""

import numpy as np
from typing import Dict, Any
import structlog

from .exceptions import NormalizationError

# Initialize structured logger
logger = structlog.get_logger(__name__)


def normalize_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Normalize a single feature vector to target dimension using simple padding/truncation.

    This function directly adjusts the vector dimension without requiring multiple
    samples or learning algorithms like PCA. It uses truncation for oversized vectors
    and zero-padding for undersized vectors.

    Parameters
    ----------
    vector : np.ndarray
        Input feature vector to normalize.
    target_dim : int
        Target dimension for the normalized vector.

    Returns
    -------
    np.ndarray
        Normalized feature vector with target dimension.

    Raises
    ------
    NormalizationError
        If normalization fails.

    Examples
    --------
    >>> raw_features = np.random.randn(1000)
    >>> normalized = normalize_vector(raw_features, 512)
    >>> assert normalized.shape == (512,)
    """
    logger.debug(
        "Normalizing single vector", input_shape=vector.shape, target_dim=target_dim
    )

    if not isinstance(vector, np.ndarray):
        raise NormalizationError(
            f"Input must be numpy array, got {type(vector)}",
            vector_shape=getattr(vector, "shape", (0,)),
            target_dimension=target_dim,
        )

    if vector.ndim != 1:
        raise NormalizationError(
            f"Input must be 1D array, got {vector.ndim}D",
            vector_shape=vector.shape,
            target_dimension=target_dim,
        )

    if len(vector) == 0:
        raise NormalizationError(
            "Cannot normalize empty vector",
            vector_shape=vector.shape,
            target_dimension=target_dim,
        )

    if target_dim <= 0:
        raise NormalizationError(
            f"Target dimension must be positive, got {target_dim}",
            vector_shape=vector.shape,
            target_dimension=target_dim,
        )

    try:
        # Simple direct normalization without PCA learning
        if len(vector) >= target_dim:
            # Truncate if vector is too long
            normalized = vector[:target_dim]
        else:
            # Zero-pad if vector is too short
            normalized = np.zeros(target_dim, dtype=vector.dtype)
            normalized[: len(vector)] = vector

        # Convert to float32 for consistency
        return normalized.astype(np.float32)

    except Exception as e:
        raise NormalizationError(
            f"Unexpected error during normalization: {str(e)}",
            vector_shape=vector.shape,
            target_dimension=target_dim,
        )


def get_normalization_statistics(vectors: list[np.ndarray]) -> Dict[str, Any]:
    """
    Calculate normalization statistics for a set of vectors.

    Parameters
    ----------
    vectors : list[np.ndarray]
        List of feature vectors to analyze.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing normalization statistics.

    Examples
    --------
    >>> vectors = [np.random.randn(100) for _ in range(10)]
    >>> stats = get_normalization_statistics(vectors)
    >>> print(f"Mean dimension: {stats['mean_dimension']}")
    """
    if not vectors:
        return {"error": "No vectors provided"}

    dimensions = [len(v) for v in vectors]
    vector_matrix = np.vstack(vectors) if len(set(dimensions)) == 1 else None

    stats = {
        "n_vectors": len(vectors),
        "dimensions": {
            "min": min(dimensions),
            "max": max(dimensions),
            "mean": np.mean(dimensions),
            "std": np.std(dimensions),
        },
        "consistent_dimensions": len(set(dimensions)) == 1,
    }

    if vector_matrix is not None:
        stats["value_statistics"] = {
            "mean": float(np.mean(vector_matrix)),
            "std": float(np.std(vector_matrix)),
            "min": float(np.min(vector_matrix)),
            "max": float(np.max(vector_matrix)),
        }

    return stats
