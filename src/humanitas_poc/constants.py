"""
Constants and configuration parameters for the HUMANITAS POC system.

This module centralizes all system parameters to enable easy tuning and
reproducible academic experiments. All constants are carefully chosen
based on established biometric research standards.
"""

from typing import Dict, Final

# =============================================================================
# Feature Dimensions
# =============================================================================

# Fingerprint feature vector dimension after normalization
FP_FEATURE_DIM: Final[int] = 512

# Face feature vector dimension (face_recognition library standard)
FACE_FEATURE_DIM: Final[int] = 128

# Final fused template dimension after feature hashing
FUSED_FEATURE_DIM: Final[int] = 1024

# =============================================================================
# Fusion Parameters
# =============================================================================

# Default fusion weights for multimodal combination
# These weights can be modified for ablation studies
DEFAULT_FUSION_WEIGHTS: Final[Dict[str, float]] = {"fingerprint": 0.7, "face": 0.3}

# Number of fingerprint samples per person (NIST SD-302 standard)
FINGERPRINTS_PER_PERSON: Final[int] = 10

# =============================================================================
# Cryptographic Parameters (Argon2)
# =============================================================================

# Argon2 time cost parameter (number of iterations)
ARGON2_TIME_COST: Final[int] = 3

# Argon2 memory cost parameter in KB (64 MB)
ARGON2_MEMORY_COST: Final[int] = 65536

# Argon2 parallelism parameter (number of threads)
ARGON2_PARALLELISM: Final[int] = 1

# Length of the final hash output in bytes
ARGON2_HASH_LENGTH: Final[int] = 32

# Length of the cryptographic salt in bytes
ARGON2_SALT_LENGTH: Final[int] = 16

# =============================================================================
# Biometric Quality Thresholds
# =============================================================================

# Minimum acceptable face quality score (0.0 to 1.0)
MIN_FACE_QUALITY_SCORE: Final[float] = 0.8

# Minimum acceptable fingerprint quality score
MIN_FP_QUALITY_SCORE: Final[float] = 70.0

# =============================================================================
# Feature Extraction Parameters
# =============================================================================

# Maximum number of ORB keypoints to detect per fingerprint
MAX_ORB_FEATURES: Final[int] = 500

# ORB scale factor between pyramid levels
ORB_SCALE_FACTOR: Final[float] = 1.2

# Number of pyramid levels for ORB
ORB_N_LEVELS: Final[int] = 8

# =============================================================================
# Test Configuration
# =============================================================================

# Default number of comparison pairs for FMR/FNMR testing
DEFAULT_TEST_PAIRS: Final[int] = 1000

# Number of noise iterations for FNMR testing
FNMR_NOISE_ITERATIONS: Final[int] = 5

# Gaussian noise standard deviation for FNMR testing
NOISE_STD_DEV: Final[float] = 0.1

# =============================================================================
# Performance Benchmarking
# =============================================================================

# Number of iterations for performance benchmarking
BENCHMARK_ITERATIONS: Final[int] = 100

# Timeout for individual operations (seconds)
OPERATION_TIMEOUT: Final[int] = 30

# =============================================================================
# ZK-Proof Parameters
# =============================================================================

# Constraint system parameters
ZK_CONSTRAINT_SYSTEM_SIZE: Final[int] = 2048

# Proof size limit in bytes
MAX_PROOF_SIZE: Final[int] = 1024

# =============================================================================
# File and Directory Constants
# =============================================================================

# Expected file extensions for fingerprint images
FINGERPRINT_EXTENSIONS: Final[tuple] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Expected file extensions for face images
FACE_EXTENSIONS: Final[tuple] = (".jpg", ".jpeg", ".png", ".bmp")

# Default output file names
DEFAULT_RESULTS_FILE: Final[str] = "humanitas_poc_results.json"
DEFAULT_LOG_FILE: Final[str] = "humanitas_poc.log"
DEFAULT_METRICS_FILE: Final[str] = "performance_metrics.json"
