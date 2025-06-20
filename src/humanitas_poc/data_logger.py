"""
Data logging and result persistence for the HUMANITAS POC system.

This module provides comprehensive data logging functionality for test results,
performance metrics, and system outputs. It generates structured data files
suitable for academic analysis, peer review, and reproducible research.

The logging system supports multiple output formats and maintains detailed
metadata for academic transparency and research reproducibility.
"""

import json
import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import gzip
import structlog
import numpy as np

from .data_models import TestResult, DatasetStatistics
from .exceptions import TestExecutionError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class TestResultLogger:
    """
    Comprehensive test result logging and persistence system.

    This class provides structured logging of test results in multiple formats
    suitable for academic analysis, data visualization, and research publication.
    It maintains detailed metadata and ensures reproducibility.

    Parameters
    ----------
    output_directory : Path
        Directory for output files.
    compress_results : bool, default=True
        Whether to compress large result files.
    include_raw_data : bool, default=False
        Whether to include raw biometric data in outputs.
    auto_backup : bool, default=True
        Whether to create backup files.

    Examples
    --------
    >>> logger = TestResultLogger(Path("./results"))
    >>> result_file = logger.log_test_results(test_results, summary)
    >>> print(f"Results saved to: {result_file}")
    """

    def __init__(
        self,
        output_directory: Path,
        compress_results: bool = True,
        include_raw_data: bool = False,
        auto_backup: bool = True,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.compress_results = compress_results
        self.include_raw_data = include_raw_data
        self.auto_backup = auto_backup

        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        self.results_dir = self.output_directory / "test_results"
        self.metrics_dir = self.output_directory / "performance_metrics"
        self.logs_dir = self.output_directory / "execution_logs"
        self.backups_dir = self.output_directory / "backups"

        for directory in [
            self.results_dir,
            self.metrics_dir,
            self.logs_dir,
            self.backups_dir,
        ]:
            directory.mkdir(exist_ok=True)

        logger.info(
            "TestResultLogger initialized",
            output_directory=str(self.output_directory),
            compress_results=compress_results,
            include_raw_data=include_raw_data,
            auto_backup=auto_backup,
        )

    def _generate_filename(
        self, base_name: str, extension: str = ".json", include_timestamp: bool = True
    ) -> str:
        """
        Generate a unique filename with timestamp.

        Parameters
        ----------
        base_name : str
            Base name for the file.
        extension : str, default=".json"
            File extension.
        include_timestamp : bool, default=True
            Whether to include timestamp in filename.

        Returns
        -------
        str
            Generated filename.
        """
        if include_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}{extension}"
        else:
            return f"{base_name}{extension}"

    def _backup_existing_file(self, file_path: Path) -> Optional[Path]:
        """
        Create backup of existing file.

        Parameters
        ----------
        file_path : Path
            Path to file that should be backed up.

        Returns
        -------
        Optional[Path]
            Path to backup file, or None if no backup created.
        """
        if not self.auto_backup or not file_path.exists():
            return None

        backup_name = f"{file_path.stem}_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
        backup_path = self.backups_dir / backup_name

        try:
            backup_path.write_bytes(file_path.read_bytes())
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")
            return None

    def _save_json(
        self, data: Dict[str, Any], file_path: Path, compress: bool = None
    ) -> Path:
        """
        Save data as JSON with optional compression.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to save.
        file_path : Path
            Output file path.
        compress : bool, default=None
            Whether to compress. Uses class default if None.

        Returns
        -------
        Path
            Path to saved file.
        """
        if compress is None:
            compress = self.compress_results

        # Create backup if file exists
        self._backup_existing_file(file_path)

        # Serialize data
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)

        if compress:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
            with gzip.open(compressed_path, "wt", encoding="utf-8") as f:
                f.write(json_str)
            logger.debug(f"Saved compressed JSON: {compressed_path}")
            return compressed_path
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.debug(f"Saved JSON: {file_path}")
            return file_path

    def _save_csv(self, data: List[Dict[str, Any]], file_path: Path) -> Path:
        """
        Save data as CSV.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of dictionaries to save as CSV.
        file_path : Path
            Output file path.

        Returns
        -------
        Path
            Path to saved file.
        """
        if not data:
            logger.warning("No data provided for CSV export")
            return file_path

        # Create backup if file exists
        self._backup_existing_file(file_path)

        # Get all unique keys from all dictionaries
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())

        fieldnames = sorted(all_keys)

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for item in data:
                # Convert complex objects to strings
                row = {}
                for key in fieldnames:
                    value = item.get(key, "")
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value, default=str)
                    else:
                        row[key] = str(value) if value is not None else ""
                writer.writerow(row)

        logger.debug(f"Saved CSV: {file_path}")
        return file_path

    def _save_pickle(self, data: Any, file_path: Path, compress: bool = None) -> Path:
        """
        Save data as pickle with optional compression.

        Parameters
        ----------
        data : Any
            Data to save.
        file_path : Path
            Output file path.
        compress : bool, default=None
            Whether to compress. Uses class default if None.

        Returns
        -------
        Path
            Path to saved file.
        """
        if compress is None:
            compress = self.compress_results

        # Create backup if file exists
        self._backup_existing_file(file_path)

        if compress:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
            with gzip.open(compressed_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved compressed pickle: {compressed_path}")
            return compressed_path
        else:
            with open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved pickle: {file_path}")
            return file_path

    def log_test_results(
        self,
        test_results: List[TestResult],
        summary_data: Optional[Dict[str, Any]] = None,
        output_formats: Optional[List[str]] = None,
    ) -> Path:
        """
        Log comprehensive test results in multiple formats.

        This is the main method for persisting test results. It saves results
        in multiple formats for different analysis needs and maintains detailed
        metadata for academic reproducibility.

        Parameters
        ----------
        test_results : List[TestResult]
            Individual test results to log.
        summary_data : Optional[Dict[str, Any]], default=None
            Summary data and metadata.
        output_formats : Optional[List[str]], default=None
            Output formats to generate. Options: 'json', 'csv', 'pickle'.
            Defaults to ['json', 'csv'].

        Returns
        -------
        Path
            Path to the primary results file (JSON).

        Raises
        ------
        TestExecutionError
            If logging fails critically.

        Examples
        --------
        >>> logger = TestResultLogger(Path("./results"))
        >>> results_file = logger.log_test_results(test_results, summary)
        >>> print(f"Results saved to: {results_file}")
        """
        if output_formats is None:
            output_formats = ["json", "csv"]

        logger.info(
            "Starting test result logging",
            n_test_results=len(test_results),
            output_formats=output_formats,
            include_summary=summary_data is not None,
        )

        try:
            # Generate timestamp for this logging session
            timestamp = datetime.now(timezone.utc)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

            # Prepare comprehensive data structure
            comprehensive_data = {
                "metadata": {
                    "logging_timestamp": timestamp.isoformat(),
                    "humanitas_poc_version": "1.0.0",
                    "total_test_results": len(test_results),
                    "data_format_version": "1.0",
                    "compression_enabled": self.compress_results,
                },
                "summary": summary_data or {},
                "individual_results": [result.to_dict() for result in test_results],
                "statistics": self._calculate_result_statistics(test_results),
            }

            # Save in JSON format (primary format)
            primary_file = None
            if "json" in output_formats:
                json_filename = self._generate_filename("test_results", ".json")
                json_path = self.results_dir / json_filename
                primary_file = self._save_json(comprehensive_data, json_path)

            # Save individual results as CSV for analysis
            if "csv" in output_formats:
                csv_filename = self._generate_filename("individual_results", ".csv")
                csv_path = self.results_dir / csv_filename

                # Convert TestResult objects to flat dictionaries for CSV
                csv_data = []
                for result in test_results:
                    result_dict = result.to_dict()
                    # Flatten nested dictionaries
                    flat_dict = self._flatten_dict(result_dict)
                    csv_data.append(flat_dict)

                self._save_csv(csv_data, csv_path)

            # Save as pickle for Python analysis
            if "pickle" in output_formats:
                pickle_filename = self._generate_filename("test_results", ".pkl")
                pickle_path = self.results_dir / pickle_filename

                # Save both raw objects and processed data
                pickle_data = {
                    "test_results_objects": test_results,
                    "comprehensive_data": comprehensive_data,
                }
                self._save_pickle(pickle_data, pickle_path)

            # Save summary data separately if provided
            if summary_data:
                summary_filename = self._generate_filename("test_summary", ".json")
                summary_path = self.results_dir / summary_filename
                self._save_json(
                    summary_data, summary_path, compress=False
                )  # Don't compress summaries

            # Generate analysis-friendly exports
            self._generate_analysis_exports(test_results, timestamp_str)

            # Log completion
            primary_result_file = (
                primary_file
                or self.results_dir / self._generate_filename("test_results", ".json")
            )

            logger.info(
                "Test result logging completed successfully",
                primary_file=str(primary_result_file),
                formats_saved=output_formats,
                total_files_created=len(output_formats) + (2 if summary_data else 1),
            )

            return primary_result_file

        except Exception as e:
            logger.error(
                "Test result logging failed",
                error=str(e),
                error_type=type(e).__name__,
                n_results=len(test_results),
            )
            raise TestExecutionError(
                f"Failed to log test results: {str(e)}",
                test_type="logging",
                test_stage="result_persistence",
            )

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary for CSV export.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary to flatten.
        parent_key : str, default=''
            Parent key prefix.
        sep : str, default='_'
            Separator for nested keys.

        Returns
        -------
        Dict[str, Any]
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v, default=str)))
            else:
                items.append((new_key, v))

        return dict(items)

    def _calculate_result_statistics(
        self, test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from test results.

        Parameters
        ----------
        test_results : List[TestResult]
            Test results to analyze.

        Returns
        -------
        Dict[str, Any]
            Statistical summary of results.
        """
        if not test_results:
            return {"error": "No test results to analyze"}

        # Basic counts
        total_tests = len(test_results)
        correct_tests = sum(1 for r in test_results if r.is_correct)
        error_tests = sum(1 for r in test_results if r.has_error)

        # Test type breakdown
        test_types = {}
        for result in test_results:
            test_type = result.test_type
            if test_type not in test_types:
                test_types[test_type] = {"total": 0, "correct": 0, "errors": 0}

            test_types[test_type]["total"] += 1
            if result.is_correct:
                test_types[test_type]["correct"] += 1
            if result.has_error:
                test_types[test_type]["errors"] += 1

        # Timing statistics
        processing_times = [
            r.total_processing_time for r in test_results if r.total_processing_time > 0
        ]
        confidence_scores = [r.confidence_score for r in test_results]

        statistics = {
            "total_tests": total_tests,
            "correct_tests": correct_tests,
            "error_tests": error_tests,
            "accuracy": correct_tests / total_tests if total_tests > 0 else 0.0,
            "error_rate": error_tests / total_tests if total_tests > 0 else 0.0,
            "test_type_breakdown": test_types,
            "timing_statistics": {
                "mean_processing_time_ms": (
                    float(np.mean(processing_times)) if processing_times else 0.0
                ),
                "std_processing_time_ms": (
                    float(np.std(processing_times)) if processing_times else 0.0
                ),
                "min_processing_time_ms": (
                    float(np.min(processing_times)) if processing_times else 0.0
                ),
                "max_processing_time_ms": (
                    float(np.max(processing_times)) if processing_times else 0.0
                ),
                "total_processing_time_ms": (
                    float(np.sum(processing_times)) if processing_times else 0.0
                ),
            },
            "confidence_statistics": {
                "mean_confidence": (
                    float(np.mean(confidence_scores)) if confidence_scores else 0.0
                ),
                "std_confidence": (
                    float(np.std(confidence_scores)) if confidence_scores else 0.0
                ),
                "min_confidence": (
                    float(np.min(confidence_scores)) if confidence_scores else 0.0
                ),
                "max_confidence": (
                    float(np.max(confidence_scores)) if confidence_scores else 0.0
                ),
            },
        }

        return statistics

    def _generate_analysis_exports(
        self, test_results: List[TestResult], timestamp_str: str
    ) -> None:
        """
        Generate additional files for data analysis.

        Parameters
        ----------
        test_results : List[TestResult]
            Test results to process.
        timestamp_str : str
            Timestamp string for filenames.
        """
        try:
            # Export timing data for performance analysis
            timing_data = []
            for result in test_results:
                timing_row = {
                    "test_id": result.test_id,
                    "test_type": result.test_type,
                    "total_time_ms": result.total_processing_time,
                    "is_correct": result.is_correct,
                    "confidence_score": result.confidence_score,
                }
                timing_row.update(result.processing_times)
                timing_data.append(timing_row)

            timing_csv_path = self.metrics_dir / f"timing_analysis_{timestamp_str}.csv"
            self._save_csv(timing_data, timing_csv_path)

            # Export accuracy data by test type
            accuracy_data = []
            test_types = set(r.test_type for r in test_results)

            for test_type in test_types:
                type_results = [r for r in test_results if r.test_type == test_type]
                accuracy_row = {
                    "test_type": test_type,
                    "total_tests": len(type_results),
                    "correct_tests": sum(1 for r in type_results if r.is_correct),
                    "accuracy": sum(1 for r in type_results if r.is_correct)
                    / len(type_results),
                    "avg_confidence": np.mean(
                        [r.confidence_score for r in type_results]
                    ),
                    "avg_processing_time_ms": np.mean(
                        [r.total_processing_time for r in type_results]
                    ),
                }
                accuracy_data.append(accuracy_row)

            accuracy_csv_path = (
                self.metrics_dir / f"accuracy_by_type_{timestamp_str}.csv"
            )
            self._save_csv(accuracy_data, accuracy_csv_path)

            logger.debug("Generated analysis export files")

        except Exception as e:
            logger.warning(f"Failed to generate analysis exports: {str(e)}")

    def log_performance_metrics(
        self, performance_data: Dict[str, Any], component_name: str = "system"
    ) -> Path:
        """
        Log performance metrics separately from test results.

        Parameters
        ----------
        performance_data : Dict[str, Any]
            Performance metrics to log.
        component_name : str, default="system"
            Name of the component being benchmarked.

        Returns
        -------
        Path
            Path to saved performance metrics file.
        """
        logger.info(f"Logging performance metrics for {component_name}")

        metrics_filename = self._generate_filename(
            f"performance_{component_name}", ".json"
        )
        metrics_path = self.metrics_dir / metrics_filename

        metrics_data = {
            "metadata": {
                "component_name": component_name,
                "logging_timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics_version": "1.0",
            },
            "performance_data": performance_data,
        }

        return self._save_json(metrics_data, metrics_path)

    def log_dataset_statistics(self, dataset_stats: DatasetStatistics) -> Path:
        """
        Log dataset statistics for reproducibility.

        Parameters
        ----------
        dataset_stats : DatasetStatistics
            Dataset statistics to log.

        Returns
        -------
        Path
            Path to saved dataset statistics file.
        """
        logger.info("Logging dataset statistics")

        stats_filename = self._generate_filename("dataset_statistics", ".json")
        stats_path = self.results_dir / stats_filename

        return self._save_json(dataset_stats.to_dict(), stats_path)


# Convenience functions for simple usage
def log_test_results(
    test_results: List[TestResult],
    output_path: Path,
    summary_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Convenience function to log test results with default settings.

    Parameters
    ----------
    test_results : List[TestResult]
        Test results to log.
    output_path : Path
        Output directory path.
    summary_data : Optional[Dict[str, Any]], default=None
        Additional summary data.

    Returns
    -------
    Path
        Path to primary results file.

    Examples
    --------
    >>> results_file = log_test_results(test_results, Path("./results"))
    >>> print(f"Results saved to: {results_file}")
    """
    logger_instance = TestResultLogger(output_path)
    return logger_instance.log_test_results(test_results, summary_data)


# Import numpy for statistics calculations
