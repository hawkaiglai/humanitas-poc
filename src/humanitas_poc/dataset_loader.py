"""
Dataset loading functionality for the HUMANITAS POC system.
Simplified version for pre-processed, clean data.
"""

from pathlib import Path
from typing import List, Optional
import structlog

from .data_models import BiometricSample, DatasetStatistics
from .exceptions import DatasetNotFoundError

logger = structlog.get_logger(__name__)


class DatasetLoader:
    """
    Loads biometric samples from the standardized, pre-processed directory.
    This version assumes the data has already been cleaned by the `scripts/` utilities.
    """

    def __init__(self, processed_dir: Path, max_samples: Optional[int] = None):
        self.processed_dir = processed_dir
        self.max_samples = max_samples

        if not self.processed_dir.is_dir():
            raise DatasetNotFoundError(
                f"Processed data directory not found: {self.processed_dir}. "
                "Please run the `preprocess_datasets.py` script first."
            )

    def load_biometric_samples(self) -> List[BiometricSample]:
        """Loads all samples from the clean data directory."""
        logger.info(
            "Loading samples from pre-processed directory...",
            path=str(self.processed_dir),
        )

        samples = []
        person_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])

        # Limit samples if max_samples is set
        if self.max_samples and len(person_dirs) > self.max_samples:
            person_dirs = person_dirs[: self.max_samples]

        for person_dir in person_dirs:
            person_id = person_dir.name
            try:
                face_files = list(person_dir.glob("face.*"))
                fp_files = sorted(list(person_dir.glob("fp_*")))

                if not face_files or not fp_files:
                    logger.warning(
                        "Skipping incomplete directory", directory=person_dir
                    )
                    continue

                sample = BiometricSample(
                    person_id=person_id,
                    fingerprint_paths=[str(p) for p in fp_files],
                    face_path=str(face_files[0]),
                )
                samples.append(sample)
            except ValueError as e:
                logger.error(f"Error creating sample for {person_id}", error=str(e))

        if not samples:
            raise DatasetNotFoundError(
                f"No valid samples found in {self.processed_dir}"
            )

        logger.info(f"Successfully loaded {len(samples)} samples from processed data.")
        return samples

    def get_statistics(self) -> Optional[DatasetStatistics]:
        """Get basic dataset statistics."""
        return None  # Simplified for now


def load_biometric_samples(
    processed_dir: Path, limit: Optional[int] = None
) -> List[BiometricSample]:
    """Convenience function to load biometric samples."""
    loader = DatasetLoader(processed_dir=processed_dir, max_samples=limit)
    return loader.load_biometric_samples()


def get_dataset_info(processed_dir: Path) -> dict:
    """Get basic information about the processed dataset."""
    info = {
        "processed_dir": str(processed_dir),
        "exists": processed_dir.exists(),
        "people_count": 0,
        "total_files": 0,
    }

    if processed_dir.exists() and processed_dir.is_dir():
        try:
            person_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
            info["people_count"] = len(person_dirs)

            for person_dir in person_dirs:
                files = list(person_dir.iterdir())
                info["total_files"] += len(files)

        except Exception as e:
            info["error"] = str(e)

    return info
