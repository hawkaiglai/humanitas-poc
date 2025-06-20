"""
Biometric quality assessment for the HUMANITAS POC system.

This module provides comprehensive quality assessment functionality for both
fingerprint and facial biometric samples. Quality assessment is critical for
ensuring reliable feature extraction and maintaining high accuracy in the
biometric system.

The quality metrics are designed to be fast, robust, and academically reproducible,
providing both overall quality scores and detailed diagnostic information.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import face_recognition
import structlog
from scipy import ndimage

from .constants import (
    MIN_FACE_QUALITY_SCORE,
    MIN_FP_QUALITY_SCORE,
)
from .exceptions import BiometricProcessingError, QualityCheckError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class BiometricQualityAssessor:
    """
    Comprehensive biometric quality assessment engine.

    This class provides sophisticated quality assessment for multiple biometric
    modalities, using computer vision and image processing techniques to evaluate
    sample quality across various dimensions.

    Parameters
    ----------
    face_quality_threshold : float, default=MIN_FACE_QUALITY_SCORE
        Minimum acceptable face quality score.
    fingerprint_quality_threshold : float, default=MIN_FP_QUALITY_SCORE
        Minimum acceptable fingerprint quality score.
    strict_mode : bool, default=False
        Whether to use strict quality requirements.

    Examples
    --------
    >>> assessor = BiometricQualityAssessor()
    >>> face_score = assessor.assess_face_quality("/path/to/face.jpg")
    >>> fp_score = assessor.assess_fingerprint_quality("/path/to/fp.png")
    >>> print(f"Face quality: {face_score}, FP quality: {fp_score}")
    """

    def __init__(
        self,
        face_quality_threshold: float = MIN_FACE_QUALITY_SCORE,
        fingerprint_quality_threshold: float = MIN_FP_QUALITY_SCORE,
        strict_mode: bool = False,
    ) -> None:
        self.face_quality_threshold = face_quality_threshold
        self.fingerprint_quality_threshold = fingerprint_quality_threshold
        self.strict_mode = strict_mode

        logger.info(
            "BiometricQualityAssessor initialized",
            face_threshold=face_quality_threshold,
            fingerprint_threshold=fingerprint_quality_threshold,
            strict_mode=strict_mode,
        )

    def assess_face_quality(self, image_path: str) -> float:
        """
        Assess the quality of a facial image.

        This method evaluates multiple quality dimensions including:
        - Face detection confidence
        - Image sharpness (blur detection)
        - Illumination quality
        - Face pose and orientation
        - Image resolution adequacy

        Parameters
        ----------
        image_path : str
            Path to the facial image file.

        Returns
        -------
        float
            Quality score between 0.0 (poor) and 1.0 (excellent).

        Raises
        ------
        QualityCheckError
            If quality assessment fails or quality is below threshold.

        Examples
        --------
        >>> quality = assess_face_quality("/path/to/good_face.jpg")
        >>> assert quality >= 0.8  # High quality face
        """
        logger.debug("Starting face quality assessment", image_path=image_path)

        # Validate image path
        if not Path(image_path).exists():
            raise BiometricProcessingError(
                f"Face image file not found: {image_path}",
                processing_stage="quality_assessment",
                sample_id=str(Path(image_path).stem),
            )

        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            if image is None:
                raise QualityCheckError(
                    "Failed to load face image",
                    quality_score=0.0,
                    minimum_threshold=self.face_quality_threshold,
                    modality="face",
                )

            # Convert to various formats for different analyses
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray_image.shape

            # Initialize quality components
            quality_components = {}

            # 1. Face Detection Quality
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                raise QualityCheckError(
                    "No faces detected in image",
                    quality_score=0.0,
                    minimum_threshold=self.face_quality_threshold,
                    modality="face",
                )
            elif len(face_locations) > 1:
                logger.warning(f"Multiple faces detected: {len(face_locations)}")
                quality_components["multiple_faces_penalty"] = 0.8
            else:
                quality_components["multiple_faces_penalty"] = 1.0

            # Use the largest face
            face_location = max(
                face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
            )
            top, right, bottom, left = face_location

            # Face size quality
            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height
            image_area = width * height
            face_ratio = face_area / image_area

            # Ideal face should occupy 10-50% of image
            if face_ratio < 0.05:
                quality_components["face_size"] = 0.3  # Too small
            elif face_ratio > 0.7:
                quality_components["face_size"] = 0.6  # Too large
            else:
                quality_components["face_size"] = min(
                    1.0, face_ratio * 4
                )  # Optimal range

            # 2. Sharpness/Blur Assessment
            face_roi = gray_image[top:bottom, left:right]
            if face_roi.size > 0:
                # Laplacian variance for blur detection
                laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()

                # Normalize sharpness score (empirically determined thresholds)
                if laplacian_var < 50:
                    quality_components["sharpness"] = 0.1  # Very blurry
                elif laplacian_var < 100:
                    quality_components["sharpness"] = 0.5  # Somewhat blurry
                elif laplacian_var < 200:
                    quality_components["sharpness"] = 0.8  # Acceptable
                else:
                    quality_components["sharpness"] = 1.0  # Sharp
            else:
                quality_components["sharpness"] = 0.0

            # 3. Illumination Quality
            mean_brightness = np.mean(face_roi) if face_roi.size > 0 else 0
            brightness_std = np.std(face_roi) if face_roi.size > 0 else 0

            # Optimal brightness: 80-180 (out of 255)
            if mean_brightness < 30 or mean_brightness > 220:
                quality_components["illumination"] = 0.2  # Too dark/bright
            elif mean_brightness < 60 or mean_brightness > 200:
                quality_components["illumination"] = 0.6  # Suboptimal
            else:
                quality_components["illumination"] = 1.0  # Good

            # Contrast quality (higher std = better contrast)
            if brightness_std < 15:
                quality_components["contrast"] = 0.3  # Low contrast
            elif brightness_std < 30:
                quality_components["contrast"] = 0.7  # Moderate contrast
            else:
                quality_components["contrast"] = 1.0  # Good contrast

            # 4. Resolution Quality
            min_face_dimension = min(face_width, face_height)
            if min_face_dimension < 50:
                quality_components["resolution"] = 0.2  # Too low resolution
            elif min_face_dimension < 100:
                quality_components["resolution"] = 0.6  # Moderate resolution
            else:
                quality_components["resolution"] = 1.0  # Good resolution

            # 5. Face Pose Quality (using face landmarks if available)
            try:
                face_landmarks = face_recognition.face_landmarks(image, [face_location])
                if face_landmarks:
                    landmarks = face_landmarks[0]

                    # Calculate pose indicators using eye and nose positions
                    left_eye = np.array(landmarks["left_eye"])
                    right_eye = np.array(landmarks["right_eye"])
                    
                    # Eye level difference (should be minimal for frontal pose)
                    eye_level_diff = abs(
                        np.mean(left_eye[:, 1]) - np.mean(right_eye[:, 1])
                    )
                    eye_distance = np.linalg.norm(
                        np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0)
                    )

                    if eye_distance > 0:
                        pose_score = max(0.0, 1.0 - (eye_level_diff / eye_distance) * 2)
                        quality_components["pose"] = min(1.0, pose_score)
                    else:
                        quality_components["pose"] = 0.5
                else:
                    quality_components["pose"] = (
                        0.7  # Assume reasonable pose if landmarks not detected
                    )
            except Exception:
                quality_components["pose"] = 0.7  # Default if landmark detection fails

            # Combine quality components with weights
            quality_weights = {
                "multiple_faces_penalty": 0.1,
                "face_size": 0.15,
                "sharpness": 0.25,
                "illumination": 0.2,
                "contrast": 0.15,
                "resolution": 0.1,
                "pose": 0.05,
            }

            # Calculate weighted average
            total_score = sum(
                quality_components[comp] * quality_weights[comp]
                for comp in quality_components
                if comp in quality_weights
            )

            # Apply strict mode penalty if enabled
            if self.strict_mode:
                total_score *= 0.9  # 10% penalty in strict mode

            # Ensure score is in valid range
            total_score = max(0.0, min(1.0, total_score))

            logger.info(
                "Face quality assessment completed",
                image_path=image_path,
                total_score=total_score,
                components=quality_components,
                faces_detected=len(face_locations),
            )

            # Check against threshold
            if total_score < self.face_quality_threshold:
                raise QualityCheckError(
                    f"Face quality score {total_score:.3f} below threshold {self.face_quality_threshold}",
                    quality_score=total_score,
                    minimum_threshold=self.face_quality_threshold,
                    modality="face",
                )

            return total_score

        except Exception as e:
            if isinstance(e, (QualityCheckError, BiometricProcessingError)):
                raise
            else:
                raise BiometricProcessingError(
                    f"Unexpected error during face quality assessment: {str(e)}",
                    processing_stage="quality_assessment",
                    sample_id=str(Path(image_path).stem),
                )

    def assess_fingerprint_quality(self, image_path: str) -> float:
        """
        Assess the quality of a fingerprint image.

        This method evaluates fingerprint quality using multiple metrics:
        - Image sharpness and clarity
        - Ridge quality and continuity
        - Contrast and dynamic range
        - Image resolution adequacy
        - Noise levels

        Parameters
        ----------
        image_path : str
            Path to the fingerprint image file.

        Returns
        -------
        float
            Quality score between 0.0 (poor) and 100.0 (excellent).

        Raises
        ------
        QualityCheckError
            If quality assessment fails or quality is below threshold.

        Examples
        --------
        >>> quality = assess_fingerprint_quality("/path/to/good_fp.png")
        >>> assert quality >= 70.0  # High quality fingerprint
        """
        logger.debug("Starting fingerprint quality assessment", image_path=image_path)

        # Validate image path
        if not Path(image_path).exists():
            raise BiometricProcessingError(
                f"Fingerprint image file not found: {image_path}",
                processing_stage="quality_assessment",
                sample_id=str(Path(image_path).stem),
            )

        try:
            # Load image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise QualityCheckError(
                    "Failed to load fingerprint image",
                    quality_score=0.0,
                    minimum_threshold=self.fingerprint_quality_threshold,
                    modality="fingerprint",
                )

            height, width = image.shape
            quality_components = {}

            # 1. Sharpness Assessment (Laplacian variance)
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

            # Normalize to 0-100 scale (empirically determined)
            if laplacian_var < 100:
                quality_components["sharpness"] = 20.0  # Very blurry
            elif laplacian_var < 300:
                quality_components["sharpness"] = 50.0  # Moderate
            elif laplacian_var < 600:
                quality_components["sharpness"] = 80.0  # Good
            else:
                quality_components["sharpness"] = 100.0  # Excellent

            # 2. Contrast and Dynamic Range
            contrast = image.std()
            dynamic_range = int(image.max()) - int(image.min())

            # Good fingerprints should have high contrast
            if contrast < 20:
                quality_components["contrast"] = 20.0
            elif contrast < 40:
                quality_components["contrast"] = 60.0
            else:
                quality_components["contrast"] = 100.0

            # Dynamic range assessment
            if dynamic_range < 100:
                quality_components["dynamic_range"] = 30.0
            elif dynamic_range < 150:
                quality_components["dynamic_range"] = 70.0
            else:
                quality_components["dynamic_range"] = 100.0

            # 3. Ridge Quality Assessment
            # Use Gabor filters to assess ridge quality
            try:
                # Apply Gabor filter bank
                gabor_responses = []
                for angle in [0, 45, 90, 135]:  # Different orientations
                    gabor_kernel = cv2.getGaborKernel(
                        (21, 21),
                        5,
                        np.radians(angle),
                        2 * np.pi * 0.1,
                        0.5,
                        0,
                        ktype=cv2.CV_32F,
                    )
                    gabor_response = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
                    gabor_responses.append(gabor_response.std())

                # Ridge quality based on maximum Gabor response
                max_gabor_response = max(gabor_responses)
                if max_gabor_response < 10:
                    quality_components["ridge_quality"] = 20.0
                elif max_gabor_response < 20:
                    quality_components["ridge_quality"] = 60.0
                else:
                    quality_components["ridge_quality"] = 100.0

            except Exception:
                # Fallback ridge assessment using local standard deviation
                kernel = np.ones((5, 5)) / 25
                local_std = ndimage.generic_filter(
                    image.astype(float), np.std, footprint=kernel
                )
                mean_local_std = np.mean(local_std)

                if mean_local_std < 5:
                    quality_components["ridge_quality"] = 30.0
                elif mean_local_std < 15:
                    quality_components["ridge_quality"] = 70.0
                else:
                    quality_components["ridge_quality"] = 100.0

            # 4. Resolution Quality
            # Fingerprints should have sufficient resolution for ridge details
            min_dimension = min(width, height)
            if min_dimension < 200:
                quality_components["resolution"] = 20.0  # Too low
            elif min_dimension < 300:
                quality_components["resolution"] = 60.0  # Moderate
            else:
                quality_components["resolution"] = 100.0  # Good

            # 5. Noise Assessment
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(image, 5)
            noise_estimate = np.mean(
                np.abs(image.astype(float) - median_filtered.astype(float))
            )

            if noise_estimate > 15:
                quality_components["noise"] = 20.0  # High noise
            elif noise_estimate > 8:
                quality_components["noise"] = 60.0  # Moderate noise
            else:
                quality_components["noise"] = 100.0  # Low noise

            # 6. Edge Density (indicator of ridge presence)
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.count_nonzero(edges) / (width * height)

            if edge_density < 0.1:
                quality_components["edge_density"] = 30.0  # Few ridges
            elif edge_density < 0.2:
                quality_components["edge_density"] = 70.0  # Moderate ridges
            else:
                quality_components["edge_density"] = 100.0  # Rich ridge structure

            # Combine quality components with weights
            quality_weights = {
                "sharpness": 0.25,
                "contrast": 0.2,
                "dynamic_range": 0.15,
                "ridge_quality": 0.25,
                "resolution": 0.05,
                "noise": 0.05,
                "edge_density": 0.05,
            }

            # Calculate weighted average
            total_score = sum(
                quality_components[comp] * quality_weights[comp]
                for comp in quality_components
                if comp in quality_weights
            )

            # Apply strict mode penalty if enabled
            if self.strict_mode:
                total_score *= 0.9  # 10% penalty in strict mode

            # Ensure score is in valid range
            total_score = max(0.0, min(100.0, total_score))

            logger.info(
                "Fingerprint quality assessment completed",
                image_path=image_path,
                total_score=total_score,
                components=quality_components,
                image_size=(width, height),
            )

            # Check against threshold
            if total_score < self.fingerprint_quality_threshold:
                raise QualityCheckError(
                    f"Fingerprint quality score {total_score:.1f} below threshold {self.fingerprint_quality_threshold}",
                    quality_score=total_score,
                    minimum_threshold=self.fingerprint_quality_threshold,
                    modality="fingerprint",
                )

            return total_score

        except Exception as e:
            if isinstance(e, (QualityCheckError, BiometricProcessingError)):
                raise
            else:
                raise BiometricProcessingError(
                    f"Unexpected error during fingerprint quality assessment: {str(e)}",
                    processing_stage="quality_assessment",
                    sample_id=str(Path(image_path).stem),
                )


# Convenience functions for standalone use
def assess_face_quality(image_path: str) -> float:
    """
    Convenience function to assess face image quality.

    Parameters
    ----------
    image_path : str
        Path to the face image file.

    Returns
    -------
    float
        Quality score between 0.0 and 1.0.

    Raises
    ------
    QualityCheckError
        If quality is below threshold or assessment fails.

    Examples
    --------
    >>> quality = assess_face_quality("/path/to/face.jpg")
    >>> print(f"Face quality: {quality:.3f}")
    """
    assessor = BiometricQualityAssessor()
    return assessor.assess_face_quality(image_path)


def assess_fingerprint_quality(image_path: str) -> float:
    """
    Convenience function to assess fingerprint image quality.

    Parameters
    ----------
    image_path : str
        Path to the fingerprint image file.

    Returns
    -------
    float
        Quality score between 0.0 and 100.0.

    Raises
    ------
    QualityCheckError
        If quality is below threshold or assessment fails.

    Examples
    --------
    >>> quality = assess_fingerprint_quality("/path/to/fingerprint.png")
    >>> print(f"Fingerprint quality: {quality:.1f}")
    """
    assessor = BiometricQualityAssessor()
    return assessor.assess_fingerprint_quality(image_path)


def assess_sample_quality(
    fingerprint_paths: List[str], face_path: str
) -> Dict[str, Any]:
    """
    Assess quality for a complete biometric sample.

    This function evaluates the quality of all biometric modalities
    for a single person and provides comprehensive quality metrics.

    Parameters
    ----------
    fingerprint_paths : List[str]
        List of paths to fingerprint images.
    face_path : str
        Path to the face image.

    Returns
    -------
    Dict[str, Any]
        Comprehensive quality assessment results.

    Examples
    --------
    >>> fp_paths = ["/path/fp1.png", "/path/fp2.png"]
    >>> results = assess_sample_quality(fp_paths, "/path/face.jpg")
    >>> print(f"Overall quality: {results['overall_quality']}")
    """
    logger.info(
        "Starting comprehensive sample quality assessment",
        n_fingerprints=len(fingerprint_paths),
        face_path=face_path,
    )

    assessor = BiometricQualityAssessor()
    results = {
        "fingerprint_qualities": [],
        "face_quality": None,
        "quality_flags": [],
        "overall_quality": 0.0,
    }

    # Assess fingerprint qualities
    fp_qualities = []
    for i, fp_path in enumerate(fingerprint_paths):
        try:
            quality = assessor.assess_fingerprint_quality(fp_path)
            fp_qualities.append(quality)
        except QualityCheckError as e:
            logger.warning(f"Fingerprint {i+1} failed quality check", error=str(e))
            results["quality_flags"].append(f"fingerprint_{i+1}_low_quality")
            fp_qualities.append(e.quality_score)

    results["fingerprint_qualities"] = fp_qualities

    # Assess face quality
    try:
        face_quality = assessor.assess_face_quality(face_path)
        results["face_quality"] = face_quality
    except QualityCheckError as e:
        logger.warning("Face failed quality check", error=str(e))
        results["quality_flags"].append("face_low_quality")
        results["face_quality"] = e.quality_score

    # Calculate overall quality
    if fp_qualities and results["face_quality"] is not None:
        # Normalize fingerprint scores to 0-1 scale for combination
        normalized_fp_qualities = [q / 100.0 for q in fp_qualities]
        avg_fp_quality = np.mean(normalized_fp_qualities)

        # Weighted combination (fingerprint 70%, face 30%)
        overall_quality = 0.7 * avg_fp_quality + 0.3 * results["face_quality"]
        results["overall_quality"] = overall_quality

        # Additional summary statistics
        results["summary"] = {
            "avg_fingerprint_quality": float(np.mean(fp_qualities)),
            "min_fingerprint_quality": (
                float(np.min(fp_qualities)) if fp_qualities else 0.0
            ),
            "max_fingerprint_quality": (
                float(np.max(fp_qualities)) if fp_qualities else 0.0
            ),
            "fingerprint_quality_std": (
                float(np.std(fp_qualities)) if len(fp_qualities) > 1 else 0.0
            ),
            "face_quality": results["face_quality"],
            "overall_quality": overall_quality,
            "quality_flags": results["quality_flags"],
        }

    logger.info(
        "Sample quality assessment completed",
        overall_quality=results["overall_quality"],
        n_quality_flags=len(results["quality_flags"]),
    )

    return results
