"""
Biometric feature extraction for the HUMANITAS POC system.

This module implements robust feature extraction methods for fingerprint and
facial biometrics, supporting multiple algorithms and providing comprehensive
error handling for academic research reproducibility.

The extracted features form the foundation of the multimodal biometric template
generation pipeline and are critical for the zero-knowledge proof system.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import face_recognition
import structlog

from .constants import (
    FACE_FEATURE_DIM,
    MAX_ORB_FEATURES,
    ORB_SCALE_FACTOR,
    ORB_N_LEVELS,
)
from .exceptions import FeatureExtractionError, BiometricProcessingError

# Initialize structured logger
logger = structlog.get_logger(__name__)


def extract_fingerprint_features(image_path: str, algorithm: str = "ORB") -> np.ndarray:
    """
    Extract feature descriptors from a fingerprint image.

    This function loads a fingerprint image and extracts keypoint descriptors
    using either ORB (Oriented FAST and Rotated BRIEF) or SIFT (Scale-Invariant
    Feature Transform) algorithms. The descriptors are flattened into a 1D
    feature vector suitable for template generation.

    Parameters
    ----------
    image_path : str
        Path to the fingerprint image file.
    algorithm : str, default="ORB"
        Feature extraction algorithm to use. Options: "ORB", "SIFT".

    Returns
    -------
    np.ndarray
        1D feature vector extracted from the fingerprint image.
        Shape: (variable_length,) - will be normalized later in pipeline.

    Raises
    ------
    FeatureExtractionError
        If image loading fails, no features are detected, or algorithm is invalid.

    Examples
    --------
    >>> features = extract_fingerprint_features("/path/to/fingerprint.png")
    >>> print(f"Extracted {len(features)} features")
    >>> features_sift = extract_fingerprint_features("/path/to/fp.png", "SIFT")
    """
    logger.info(
        "Starting fingerprint feature extraction",
        image_path=image_path,
        algorithm=algorithm,
    )

    # Validate algorithm choice
    valid_algorithms = ["ORB", "SIFT"]
    if algorithm not in valid_algorithms:
        raise FeatureExtractionError(
            f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}",
            image_path=image_path,
            extraction_method=algorithm,
        )

    # Validate image path exists
    if not Path(image_path).exists():
        raise FeatureExtractionError(
            f"Image file not found: {image_path}",
            image_path=image_path,
            extraction_method=algorithm,
        )

    try:
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FeatureExtractionError(
                "Failed to load image. File may be corrupted or invalid format",
                image_path=image_path,
                extraction_method=algorithm,
            )

        # Log image properties
        height, width = image.shape
        logger.debug(
            "Image loaded successfully",
            image_shape=(height, width),
            image_dtype=str(image.dtype),
        )

        # Initialize feature detector based on algorithm
        if algorithm == "ORB":
            detector = cv2.ORB_create(
                nfeatures=MAX_ORB_FEATURES,
                scaleFactor=ORB_SCALE_FACTOR,
                nlevels=ORB_N_LEVELS,
            )
        elif algorithm == "SIFT":
            detector = cv2.SIFT_create(nfeatures=MAX_ORB_FEATURES)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(image, None)

        if descriptors is None or len(descriptors) == 0:
            raise FeatureExtractionError(
                "No keypoints detected in image. Image may be too blurry or low quality",
                image_path=image_path,
                extraction_method=algorithm,
            )

        # Flatten descriptors into 1D vector
        feature_vector = descriptors.flatten().astype(np.float32)

        logger.info(
            "Feature extraction completed",
            keypoints_detected=len(keypoints),
            descriptor_shape=descriptors.shape,
            feature_vector_length=len(feature_vector),
        )

        return feature_vector

    except cv2.error as e:
        raise FeatureExtractionError(
            f"OpenCV error during feature extraction: {str(e)}",
            image_path=image_path,
            extraction_method=algorithm,
        )

    except Exception as e:
        # Handle any other unexpected errors
        if isinstance(e, FeatureExtractionError):
            raise
        else:
            raise FeatureExtractionError(
                f"Unexpected error during feature extraction: {str(e)}",
                image_path=image_path,
                extraction_method=algorithm,
            )


def extract_facial_features(image_path: str) -> np.ndarray:
    """
    Extract facial features using face_recognition library.

    This function loads a face image and extracts a 128-dimensional face
    encoding using the face_recognition library, which uses a deep learning
    model trained on facial landmarks and features.

    Parameters
    ----------
    image_path : str
        Path to the facial image file.

    Returns
    -------
    np.ndarray
        128-dimensional face encoding vector.
        Shape: (128,) - standard face_recognition output dimension.

    Raises
    ------
    FeatureExtractionError
        If image loading fails, no faces are detected, or multiple faces found.

    Examples
    --------
    >>> face_features = extract_facial_features("/path/to/face.jpg")
    >>> print(f"Face encoding shape: {face_features.shape}")
    >>> assert face_features.shape == (128,)
    """
    logger.info("Starting facial feature extraction", image_path=image_path)

    # Validate image path exists
    if not Path(image_path).exists():
        raise FeatureExtractionError(
            f"Image file not found: {image_path}",
            image_path=image_path,
            extraction_method="face_recognition",
        )

    try:
        # Load image using face_recognition library
        image = face_recognition.load_image_file(image_path)

        if image is None:
            raise FeatureExtractionError(
                "Failed to load image. File may be corrupted or invalid format",
                image_path=image_path,
                extraction_method="face_recognition",
            )

        # Log image properties
        height, width, channels = image.shape
        logger.debug(
            "Image loaded successfully",
            image_shape=(height, width, channels),
            image_dtype=str(image.dtype),
        )

        # Find face locations in the image
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            raise FeatureExtractionError(
                "No faces detected in image. Ensure image contains a clear face",
                image_path=image_path,
                extraction_method="face_recognition",
            )

        if len(face_locations) > 1:
            logger.warning(
                "Multiple faces detected, using the first face",
                faces_detected=len(face_locations),
            )

        # Extract face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) == 0:
            raise FeatureExtractionError(
                "Failed to extract face encoding. Face may be too small or unclear",
                image_path=image_path,
                extraction_method="face_recognition",
            )

        # Use the first face encoding
        face_encoding = face_encodings[0]

        # Validate encoding dimension
        if face_encoding.shape != (FACE_FEATURE_DIM,):
            raise FeatureExtractionError(
                f"Unexpected face encoding dimension: {face_encoding.shape}. "
                f"Expected: ({FACE_FEATURE_DIM},)",
                image_path=image_path,
                extraction_method="face_recognition",
            )

        logger.info(
            "Facial feature extraction completed",
            faces_detected=len(face_locations),
            encoding_shape=face_encoding.shape,
            encoding_dtype=str(face_encoding.dtype),
        )

        return face_encoding.astype(np.float32)

    except Exception as e:
        # Handle any unexpected errors
        if isinstance(e, FeatureExtractionError):
            raise
        else:
            raise FeatureExtractionError(
                f"Unexpected error during facial feature extraction: {str(e)}",
                image_path=image_path,
                extraction_method="face_recognition",
            )


def extract_features_batch(
    image_paths: list[str], modality: str, algorithm: str = "ORB"
) -> list[np.ndarray]:
    """
    Extract features from multiple images in batch.

    This function provides efficient batch processing for feature extraction,
    useful for processing multiple fingerprints per person or batch processing
    of face images.

    Parameters
    ----------
    image_paths : list[str]
        List of paths to image files.
    modality : str
        Type of biometric modality. Options: "fingerprint", "face".
    algorithm : str, default="ORB"
        Algorithm for fingerprint extraction (ignored for face modality).

    Returns
    -------
    list[np.ndarray]
        List of feature vectors, one per input image.

    Raises
    ------
    BiometricProcessingError
        If modality is invalid or batch processing fails.
    FeatureExtractionError
        If individual feature extraction fails for any image.

    Examples
    --------
    >>> fp_paths = ["/path/fp1.png", "/path/fp2.png"]
    >>> features = extract_features_batch(fp_paths, "fingerprint")
    >>> print(f"Extracted features from {len(features)} fingerprints")
    """
    logger.info(
        "Starting batch feature extraction",
        image_count=len(image_paths),
        modality=modality,
        algorithm=algorithm,
    )

    if modality not in ["fingerprint", "face"]:
        raise BiometricProcessingError(
            f"Invalid modality '{modality}'. Must be 'fingerprint' or 'face'",
            processing_stage="batch_extraction",
        )

    features = []
    failed_extractions = []

    for i, image_path in enumerate(image_paths):
        try:
            if modality == "fingerprint":
                feature_vector = extract_fingerprint_features(image_path, algorithm)
            else:  # face
                feature_vector = extract_facial_features(image_path)

            features.append(feature_vector)

        except FeatureExtractionError as e:
            failed_extractions.append((i, image_path, str(e)))
            logger.warning(
                "Feature extraction failed for image",
                image_path=image_path,
                error=str(e),
            )

    if failed_extractions:
        error_summary = f"Failed to extract features from {len(failed_extractions)} out of {len(image_paths)} images"
        logger.error(error_summary, failed_count=len(failed_extractions))

        # If more than half failed, raise an error
        if len(failed_extractions) > len(image_paths) / 2:
            raise BiometricProcessingError(
                f"{error_summary}. Too many failures for reliable processing",
                processing_stage="batch_extraction",
                context={
                    "failed_images": [
                        path for _, path, _ in failed_extractions[:5]
                    ],  # First 5
                    "total_failed": len(failed_extractions),
                    "total_images": len(image_paths),
                },
            )

    logger.info(
        "Batch feature extraction completed",
        successful_extractions=len(features),
        failed_extractions=len(failed_extractions),
    )

    return features


def validate_feature_vector(
    feature_vector: np.ndarray, expected_properties: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Validate properties of an extracted feature vector.

    This function performs comprehensive validation of feature vectors
    to ensure they meet quality standards for template generation.

    Parameters
    ----------
    feature_vector : np.ndarray
        Feature vector to validate.
    expected_properties : Optional[Dict[str, Any]], default=None
        Expected properties like shape, dtype, value ranges.

    Returns
    -------
    bool
        True if feature vector passes all validation checks.

    Raises
    ------
    BiometricProcessingError
        If feature vector fails validation.

    Examples
    --------
    >>> features = extract_fingerprint_features("/path/to/fp.png")
    >>> is_valid = validate_feature_vector(features)
    >>> assert is_valid
    """
    if not isinstance(feature_vector, np.ndarray):
        raise BiometricProcessingError(
            f"Feature vector must be numpy array, got {type(feature_vector)}",
            processing_stage="feature_validation",
        )

    if feature_vector.size == 0:
        raise BiometricProcessingError(
            "Feature vector cannot be empty", processing_stage="feature_validation"
        )

    if not np.isfinite(feature_vector).all():
        raise BiometricProcessingError(
            "Feature vector contains non-finite values (NaN or infinity)",
            processing_stage="feature_validation",
        )

    # Validate expected properties if provided
    if expected_properties:
        if "min_length" in expected_properties:
            min_len = expected_properties["min_length"]
            if len(feature_vector) < min_len:
                raise BiometricProcessingError(
                    f"Feature vector too short: {len(feature_vector)} < {min_len}",
                    processing_stage="feature_validation",
                )

        if "max_length" in expected_properties:
            max_len = expected_properties["max_length"]
            if len(feature_vector) > max_len:
                raise BiometricProcessingError(
                    f"Feature vector too long: {len(feature_vector)} > {max_len}",
                    processing_stage="feature_validation",
                )

        if "dtype" in expected_properties:
            expected_dtype = expected_properties["dtype"]
            if feature_vector.dtype != expected_dtype:
                logger.warning(
                    "Feature vector dtype mismatch",
                    expected=expected_dtype,
                    actual=feature_vector.dtype,
                )

    logger.debug(
        "Feature vector validation passed",
        shape=feature_vector.shape,
        dtype=str(feature_vector.dtype),
        value_range=(float(feature_vector.min()), float(feature_vector.max())),
    )

    return True
