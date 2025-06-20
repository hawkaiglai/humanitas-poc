"""
Configuration management for the HUMANITAS POC system.

This module handles all configuration loading from environment variables
and .env files, ensuring consistent configuration across different
deployment environments while maintaining security best practices.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# =============================================================================
# Base Paths
# =============================================================================
# Define the base directory for the project
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Project root directory (parent of src/)
PROJECT_ROOT: Path = BASE_DIR.parent

# =============================================================================
# Dataset Configuration
# =============================================================================
# Base dataset directory
DATASET_DIR: Path = PROJECT_ROOT / "data" / "raw"

# NIST Special Database 302 (Fingerprints)
NIST_FP_DATA_PATH: Path = Path(
    os.getenv("NIST_FP_DATA_PATH", str(DATASET_DIR / "nist_sd302"))
)

# Labeled Faces in the Wild (LFW) dataset
LFW_FACES_DATA_PATH: Path = Path(
    os.getenv("LFW_FACES_DATA_PATH", str(DATASET_DIR / "lfw"))
)

# Alternative face dataset path (for extended testing)
ALTERNATIVE_FACE_DATA_PATH: Optional[Path] = None
if alt_path := os.getenv("ALTERNATIVE_FACE_DATA_PATH"):
    ALTERNATIVE_FACE_DATA_PATH = Path(alt_path)

# =============================================================================
# Output and Logging Configuration
# =============================================================================
# Processed data output directory
OUTPUT_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed"

# Log files directory
OUTPUT_LOG_PATH: Path = PROJECT_ROOT / "logs"

# Results output directory
RESULTS_PATH: Path = PROJECT_ROOT / "results"

# Temporary files directory
TEMP_PATH: Path = PROJECT_ROOT / "temp"

# Processed data with HUMANITAS standard format
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed" / "humanitas_standard"

# Ensure all output directories exist
for directory in [OUTPUT_DATA_PATH, OUTPUT_LOG_PATH, RESULTS_PATH, TEMP_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Logging Configuration
# =============================================================================
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Enable structured logging output
STRUCTURED_LOGGING: bool = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"

# Log to file in addition to console
LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# Maximum log file size in MB
MAX_LOG_SIZE_MB: int = int(os.getenv("MAX_LOG_SIZE_MB", "100"))

# Number of log files to retain
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# =============================================================================
# Performance and Processing Configuration
# =============================================================================
# Maximum number of parallel processes for data processing
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4)))

# Memory limit for processing in MB (0 means unlimited)
MEMORY_LIMIT_MB: int = int(os.getenv("MEMORY_LIMIT_MB", "0"))

# Enable GPU acceleration if available
USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"

# Batch size for processing multiple samples
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))

# =============================================================================
# Testing and Validation Configuration
# =============================================================================
# Seed for random number generation (for reproducible results)
RANDOM_SEED: Optional[int] = None
if seed_str := os.getenv("RANDOM_SEED"):
    RANDOM_SEED = int(seed_str)

# Maximum number of samples to process (0 means unlimited)
MAX_SAMPLES: int = int(os.getenv("MAX_SAMPLES", "0"))

# Enable detailed performance profiling
ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "false").lower() == "true"

# Skip dataset validation for faster startup
SKIP_DATASET_VALIDATION: bool = (
    os.getenv("SKIP_DATASET_VALIDATION", "false").lower() == "true"
)

# =============================================================================
# Security and Cryptography Configuration
# =============================================================================
# Enable ZK-proof generation (can be disabled for testing)
ENABLE_ZK_PROOFS: bool = os.getenv("ENABLE_ZK_PROOFS", "true").lower() == "true"

# ZK-proof circuit compilation cache directory
ZK_CACHE_DIR: Path = PROJECT_ROOT / ".zk_cache"
ZK_CACHE_DIR.mkdir(exist_ok=True)

# Custom salt for deterministic testing (leave empty for production)
CUSTOM_SALT: Optional[str] = os.getenv("CUSTOM_SALT")

# =============================================================================
# Development and Debugging Configuration
# =============================================================================
# Enable debug mode (more verbose output, additional checks)
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Save intermediate processing results for debugging
SAVE_INTERMEDIATES: bool = os.getenv("SAVE_INTERMEDIATES", "false").lower() == "true"

# Enable memory usage monitoring
MONITOR_MEMORY: bool = os.getenv("MONITOR_MEMORY", "false").lower() == "true"

# =============================================================================
# Experimental Features Configuration
# =============================================================================
# Enable experimental fusion algorithms
ENABLE_EXPERIMENTAL_FUSION: bool = (
    os.getenv("ENABLE_EXPERIMENTAL_FUSION", "false").lower() == "true"
)

# Use alternative feature extraction methods
ALTERNATIVE_FEATURE_EXTRACTION: bool = (
    os.getenv("ALTERNATIVE_FEATURE_EXTRACTION", "false").lower() == "true"
)


# =============================================================================
# Configuration Validation
# =============================================================================
def validate_configuration() -> bool:
    """
    Validate the current configuration settings.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise.

    Raises
    ------
    ValueError
        If critical configuration parameters are invalid.
    """
    errors = []

    # Check that dataset paths exist if not skipping validation
    if not SKIP_DATASET_VALIDATION:
        if not NIST_FP_DATA_PATH.exists():
            errors.append(f"NIST fingerprint dataset not found at {NIST_FP_DATA_PATH}")

        if not LFW_FACES_DATA_PATH.exists():
            errors.append(f"LFW face dataset not found at {LFW_FACES_DATA_PATH}")

    # Validate numeric parameters
    if MAX_WORKERS < 1:
        errors.append("MAX_WORKERS must be at least 1")

    if BATCH_SIZE < 1:
        errors.append("BATCH_SIZE must be at least 1")

    if MEMORY_LIMIT_MB < 0:
        errors.append("MEMORY_LIMIT_MB cannot be negative")

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL not in valid_log_levels:
        errors.append(f"LOG_LEVEL must be one of {valid_log_levels}")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    return True


def get_config_summary() -> dict:
    """
    Get a summary of the current configuration.

    Returns
    -------
    dict
        Dictionary containing key configuration parameters.
    """
    return {
        "dataset_paths": {
            "nist_fingerprints": str(NIST_FP_DATA_PATH),
            "lfw_faces": str(LFW_FACES_DATA_PATH),
        },
        "output_paths": {
            "data": str(OUTPUT_DATA_PATH),
            "logs": str(OUTPUT_LOG_PATH),
            "results": str(RESULTS_PATH),
        },
        "processing": {
            "max_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "use_gpu": USE_GPU,
        },
        "features": {
            "enable_zk_proofs": ENABLE_ZK_PROOFS,
            "debug_mode": DEBUG_MODE,
            "enable_profiling": ENABLE_PROFILING,
        },
        "logging": {
            "level": LOG_LEVEL,
            "structured": STRUCTURED_LOGGING,
            "to_file": LOG_TO_FILE,
        },
    }


# Validate configuration on import
if not DEBUG_MODE:
    validate_configuration()
