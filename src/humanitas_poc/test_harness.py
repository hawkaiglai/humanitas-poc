"""
Test harness orchestrator for the HUMANITAS POC system.

This module provides the main TestHarness class that orchestrates all automated
testing including FMR/FNMR analysis, performance benchmarking, and system
validation. It serves as the central coordinator for academic evaluation
and empirical validation of the biometric system.

The test harness generates comprehensive results suitable for peer-reviewed
academic publications and provides detailed metrics for system evaluation.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import structlog
import numpy as np

from .data_models import BiometricSample, TestResult, DatasetStatistics
from .dataset_loader import DatasetLoader
from .feature_extraction import extract_fingerprint_features, extract_facial_features
from .fusion import BiometricFusionEngine
from .template_generator import SecureTemplateGenerator
from .zk_prover import ZkProver
from .test_fmr_fnmr import FMRFNMRTester
from .test_performance import PerformanceBenchmarker
from .data_logger import TestResultLogger
from .quality_assessment import BiometricQualityAssessor
from .exceptions import (
    TestExecutionError,
)
from .constants import DEFAULT_TEST_PAIRS, FNMR_NOISE_ITERATIONS, BENCHMARK_ITERATIONS
from .utils import timer, generate_test_id

# Initialize structured logger
logger = structlog.get_logger(__name__)


class TestHarness:
    """
    Main orchestrator for all automated biometric system tests.

    This class coordinates the execution of comprehensive tests including
    FMR/FNMR analysis, performance benchmarking, quality assessment, and
    system validation. It generates academic-quality results suitable for
    peer-reviewed publications.

    Parameters
    ----------
    dataset_loader : DatasetLoader
        Configured dataset loader for biometric samples.
    output_path : Path
        Directory for test result output.
    test_config : Optional[Dict[str, Any]], default=None
        Configuration parameters for testing.
    enable_zk_proofs : bool, default=True
        Whether to include ZK proof generation in tests.
    enable_quality_checks : bool, default=True
        Whether to perform quality assessment.

    Examples
    --------
    >>> loader = DatasetLoader(nist_path, lfw_path)
    >>> harness = TestHarness(loader, output_path)
    >>> results = harness.run_all_tests()
    >>> print(f"Completed {len(results)} tests")
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        output_path: Path,
        test_config: Optional[Dict[str, Any]] = None,
        enable_zk_proofs: bool = True,
        enable_quality_checks: bool = True,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.output_path = Path(output_path)
        self.test_config = test_config or {}
        self.enable_zk_proofs = enable_zk_proofs
        self.enable_quality_checks = enable_quality_checks

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize test components
        self.fusion_engine = BiometricFusionEngine()
        self.template_generator = SecureTemplateGenerator()
        self.zk_prover = ZkProver() if enable_zk_proofs else None
        self.quality_assessor = (
            BiometricQualityAssessor() if enable_quality_checks else None
        )

        # Initialize test modules
        self.fmr_fnmr_tester = FMRFNMRTester()
        self.performance_benchmarker = PerformanceBenchmarker()
        self.result_logger = TestResultLogger(self.output_path)

        # Test execution state
        self.test_session_id = generate_test_id()
        self.test_results: List[TestResult] = []
        self.dataset_stats: Optional[DatasetStatistics] = None
        self.processed_samples: List[BiometricSample] = []

        logger.info(
            "TestHarness initialized",
            test_session_id=self.test_session_id,
            output_path=str(self.output_path),
            enable_zk_proofs=enable_zk_proofs,
            enable_quality_checks=enable_quality_checks,
            test_config=self.test_config,
        )

    def _load_and_prepare_samples(self) -> List[BiometricSample]:
        """
        Load biometric samples and prepare them for testing.

        Returns
        -------
        List[BiometricSample]
            List of loaded and validated biometric samples.

        Raises
        ------
        TestExecutionError
            If sample loading or preparation fails.
        """
        logger.info("Loading and preparing biometric samples")

        try:
            # Load samples from datasets
            samples = self.dataset_loader.load_biometric_samples()
            self.dataset_stats = self.dataset_loader.get_statistics()

            if not samples:
                raise TestExecutionError(
                    "No biometric samples loaded from datasets",
                    test_type="preparation",
                    test_stage="sample_loading",
                )

            logger.info(f"Loaded {len(samples)} biometric samples")

            # Apply sample limit if specified in config
            max_samples = self.test_config.get("max_test_samples")
            if max_samples and len(samples) > max_samples:
                samples = samples[:max_samples]
                logger.info(f"Limited to {max_samples} samples for testing")

            return samples

        except Exception as e:
            if isinstance(e, TestExecutionError):
                raise
            else:
                raise TestExecutionError(
                    f"Failed to load and prepare samples: {str(e)}",
                    test_type="preparation",
                    test_stage="sample_loading",
                )

    @timer
    def _extract_biometric_features(
        self, samples: List[BiometricSample]
    ) -> List[BiometricSample]:
        """
        Extract features from all biometric samples.

        Parameters
        ----------
        samples : List[BiometricSample]
            Biometric samples to process.

        Returns
        -------
        List[BiometricSample]
            Samples with extracted features.
        """
        logger.info(f"Extracting features from {len(samples)} samples")

        processed_samples = []
        extraction_failures = 0

        for i, sample in enumerate(samples):
            try:
                # Extract fingerprint features - store as list of arrays
                fp_features = []
                for fp_path in sample.fingerprint_paths:
                    try:
                        features = extract_fingerprint_features(fp_path)
                        fp_features.append(features)
                    except Exception as e:
                        logger.warning(
                            "Failed to extract fingerprint features",
                            sample_id=sample.person_id,
                            fp_path=fp_path,
                            error=str(e),
                        )

                if not fp_features:
                    logger.error(
                        f"No fingerprint features extracted for {sample.person_id}"
                    )
                    extraction_failures += 1
                    continue

                # Extract face features
                try:
                    face_features = extract_facial_features(sample.face_path)
                except Exception as e:
                    logger.warning(
                        "Failed to extract face features",
                        sample_id=sample.person_id,
                        face_path=sample.face_path,
                        error=str(e),
                    )
                    extraction_failures += 1
                    continue

                # Update sample with features - store list directly
                sample.fingerprint_features = fp_features
                sample.face_features = face_features

                # Store extraction metadata
                sample.processing_metadata["feature_extraction"] = {
                    "fingerprint_count": len(fp_features),
                    "face_features_dim": len(face_features),
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                }

                processed_samples.append(sample)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(
                    f"Failed to process sample {sample.person_id}", error=str(e)
                )
                extraction_failures += 1

        success_rate = len(processed_samples) / len(samples)
        logger.info(
            "Feature extraction completed",
            successful_samples=len(processed_samples),
            failed_samples=extraction_failures,
            success_rate=success_rate,
        )

        if success_rate < 0.5:
            raise TestExecutionError(
                f"Feature extraction success rate too low: {success_rate:.2%}",
                test_type="preparation",
                test_stage="feature_extraction",
            )

        return processed_samples

    @timer
    def _generate_biometric_templates(
        self, samples: List[BiometricSample]
    ) -> List[BiometricSample]:
        """
        Generate secure biometric templates from processed samples.

        Parameters
        ----------
        samples : List[BiometricSample]
            Samples with extracted features.

        Returns
        -------
        List[BiometricSample]
            Samples with generated templates.
        """
        logger.info(f"Generating biometric templates for {len(samples)} samples")

        template_failures = 0

        for i, sample in enumerate(samples):
            try:
                # Fuse multimodal features - pass list directly
                fused_template = self.fusion_engine.fuse_features(
                    sample.fingerprint_features, sample.face_features
                )

                # Generate secure template
                template_hash, salt = self.template_generator.generate_secure_template(
                    fused_template
                )

                # Update sample
                sample.fused_template = fused_template.tobytes()
                sample.template_hash = template_hash
                sample.salt = salt

                # Store template generation metadata
                sample.processing_metadata["template_generation"] = {
                    "fusion_method": "weighted_hash",
                    "hash_algorithm": "argon2",
                    "template_length": len(template_hash),
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Generated templates for {i + 1}/{len(samples)} samples"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to generate template for {sample.person_id}", error=str(e)
                )
                template_failures += 1

        successful_templates = len(samples) - template_failures
        success_rate = successful_templates / len(samples)

        logger.info(
            "Template generation completed",
            successful_templates=successful_templates,
            failed_templates=template_failures,
            success_rate=success_rate,
        )

        if success_rate < 0.8:
            raise TestExecutionError(
                f"Template generation success rate too low: {success_rate:.2%}",
                test_type="preparation",
                test_stage="template_generation",
            )

        return samples

    def run_quality_assessment(self, samples: List[BiometricSample]) -> Dict[str, Any]:
        """
        Run comprehensive quality assessment on biometric samples.

        Parameters
        ----------
        samples : List[BiometricSample]
            Biometric samples to assess.

        Returns
        -------
        Dict[str, Any]
            Quality assessment results.
        """
        if not self.enable_quality_checks or not self.quality_assessor:
            logger.info("Quality assessment disabled, skipping")
            return {"status": "skipped", "reason": "disabled"}

        logger.info(f"Running quality assessment on {len(samples)} samples")

        quality_results = {
            "samples_assessed": 0,
            "quality_failures": 0,
            "face_quality_scores": [],
            "fingerprint_quality_scores": [],
            "overall_quality_scores": [],
            "quality_flags": [],
        }

        for sample in samples:
            try:
                # Assess sample quality
                from .quality_assessment import assess_sample_quality

                sample_quality = assess_sample_quality(
                    sample.fingerprint_paths, sample.face_path
                )

                # Store quality scores in sample
                sample.quality_scores = sample_quality["summary"]

                # Aggregate results
                quality_results["samples_assessed"] += 1
                quality_results["face_quality_scores"].append(
                    sample_quality["face_quality"]
                )
                quality_results["fingerprint_quality_scores"].extend(
                    sample_quality["fingerprint_qualities"]
                )
                quality_results["overall_quality_scores"].append(
                    sample_quality["overall_quality"]
                )
                quality_results["quality_flags"].extend(sample_quality["quality_flags"])

                if sample_quality["quality_flags"]:
                    quality_results["quality_failures"] += 1

            except Exception as e:
                logger.warning(
                    f"Quality assessment failed for {sample.person_id}", error=str(e)
                )
                quality_results["quality_failures"] += 1

        # Calculate summary statistics
        if quality_results["face_quality_scores"]:
            quality_results["face_quality_stats"] = {
                "mean": float(np.mean(quality_results["face_quality_scores"])),
                "std": float(np.std(quality_results["face_quality_scores"])),
                "min": float(np.min(quality_results["face_quality_scores"])),
                "max": float(np.max(quality_results["face_quality_scores"])),
            }

        if quality_results["fingerprint_quality_scores"]:
            quality_results["fingerprint_quality_stats"] = {
                "mean": float(np.mean(quality_results["fingerprint_quality_scores"])),
                "std": float(np.std(quality_results["fingerprint_quality_scores"])),
                "min": float(np.min(quality_results["fingerprint_quality_scores"])),
                "max": float(np.max(quality_results["fingerprint_quality_scores"])),
            }

        logger.info(
            "Quality assessment completed",
            samples_assessed=quality_results["samples_assessed"],
            quality_failures=quality_results["quality_failures"],
        )

        return quality_results

    def run_fmr_fnmr_test(self, samples: List[BiometricSample]) -> Dict[str, Any]:
        """
        Run False Match Rate and False Non-Match Rate tests.

        Parameters
        ----------
        samples : List[BiometricSample]
            Processed biometric samples.

        Returns
        -------
        Dict[str, Any]
            FMR/FNMR test results.
        """
        logger.info("Starting FMR/FNMR testing")

        try:
            # Configure test parameters
            test_params = {
                "test_pairs": self.test_config.get(
                    "fmr_test_pairs", DEFAULT_TEST_PAIRS
                ),
                "noise_iterations": self.test_config.get(
                    "fnmr_noise_iterations", FNMR_NOISE_ITERATIONS
                ),
                "enable_zk_proofs": self.enable_zk_proofs,
            }

            # Run FMR/FNMR tests
            fmr_fnmr_results = self.fmr_fnmr_tester.calculate_fmr_fnmr(
                samples, **test_params
            )

            # Convert results to TestResult objects and store
            for result_data in fmr_fnmr_results.get("detailed_results", []):
                test_result = TestResult(
                    test_id=generate_test_id(),
                    test_type=result_data["test_type"],
                    person1_id=result_data["person1_id"],
                    person2_id=result_data.get("person2_id"),
                    outcome=result_data["outcome"],
                    expected_outcome=result_data["expected_outcome"],
                    confidence_score=result_data["confidence_score"],
                    processing_times=result_data["processing_times"],
                    template_hash=result_data["template_hash"],
                    zk_proof_valid=result_data.get("zk_proof_valid", False),
                    test_parameters=test_params,
                )
                self.test_results.append(test_result)

            logger.info(
                "FMR/FNMR testing completed",
                fmr=fmr_fnmr_results.get("fmr"),
                fnmr=fmr_fnmr_results.get("fnmr"),
                total_tests=len(fmr_fnmr_results.get("detailed_results", [])),
            )

            return fmr_fnmr_results

        except Exception as e:
            raise TestExecutionError(
                f"FMR/FNMR testing failed: {str(e)}",
                test_type="fmr_fnmr",
                test_stage="execution",
            )

    def run_performance_test(self, samples: List[BiometricSample]) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmarking.

        Parameters
        ----------
        samples : List[BiometricSample]
            Processed biometric samples.

        Returns
        -------
        Dict[str, Any]
            Performance test results.
        """
        logger.info("Starting performance benchmarking")

        try:
            # Configure benchmark parameters
            benchmark_params = {
                "iterations": self.test_config.get(
                    "benchmark_iterations", BENCHMARK_ITERATIONS
                ),
                "enable_zk_proofs": self.enable_zk_proofs,
                "include_memory_profiling": self.test_config.get(
                    "profile_memory", False
                ),
            }

            # Run performance benchmarks
            performance_results = self.performance_benchmarker.benchmark_all_components(
                samples[0] if samples else None,  # Use first sample for benchmarking
                **benchmark_params,
            )

            # Create performance test results
            perf_test_result = TestResult(
                test_id=generate_test_id(),
                test_type="performance",
                person1_id=samples[0].person_id if samples else "benchmark",
                person2_id=None,
                outcome="completed",
                expected_outcome="completed",
                confidence_score=1.0,
                processing_times=performance_results.get("component_times", {}),
                template_hash=(
                    samples[0].template_hash
                    if samples and samples[0].template_hash
                    else "benchmark"
                ),
                zk_proof_valid=True,
                test_parameters=benchmark_params,
            )

            self.test_results.append(perf_test_result)

            logger.info(
                "Performance benchmarking completed",
                total_time_ms=performance_results.get("total_time_ms"),
                components_tested=len(performance_results.get("component_times", {})),
            )

            return performance_results

        except Exception as e:
            raise TestExecutionError(
                f"Performance benchmarking failed: {str(e)}",
                test_type="performance",
                test_stage="execution",
            )

    def run_system_validation(self, samples: List[BiometricSample]) -> Dict[str, Any]:
        """
        Run comprehensive system validation tests.

        Parameters
        ----------
        samples : List[BiometricSample]
            Processed biometric samples.

        Returns
        -------
        Dict[str, Any]
            System validation results.
        """
        logger.info("Starting system validation")

        validation_results = {
            "template_consistency": True,
            "zk_proof_validation": True,
            "fusion_reproducibility": True,
            "error_conditions": [],
            "validation_summary": {},
        }

        try:
            # Test template consistency
            if samples:
                sample = samples[0]
                if sample.has_template:
                    # Verify template can be regenerated consistently
                    from .template_generator import verify_secure_template

                    fused_vector = np.frombuffer(
                        sample.fused_template, dtype=np.float32
                    )

                    is_consistent = verify_secure_template(
                        fused_vector, sample.template_hash, sample.salt
                    )

                    validation_results["template_consistency"] = is_consistent
                    if not is_consistent:
                        validation_results["error_conditions"].append(
                            "template_inconsistency"
                        )

            # Test ZK proof validation if enabled
            if self.enable_zk_proofs and self.zk_prover and samples:
                try:
                    sample = samples[0]
                    if sample.has_template:
                        # Generate and verify a ZK proof
                        proving_key, verifying_key = self.zk_prover.setup()

                        fused_vector = np.frombuffer(
                            sample.fused_template, dtype=np.float32
                        )
                        private_inputs = {
                            "fused_vector": fused_vector,
                            "salt": sample.salt,
                            "template_hash": sample.template_hash,
                        }

                        proof = self.zk_prover.prove(proving_key, private_inputs)
                        public_inputs = {"template_hash": sample.template_hash}

                        is_valid = self.zk_prover.verify(
                            verifying_key, proof, public_inputs
                        )
                        validation_results["zk_proof_validation"] = is_valid

                        if not is_valid:
                            validation_results["error_conditions"].append(
                                "zk_proof_invalid"
                            )

                except Exception as e:
                    logger.warning(f"ZK proof validation failed: {str(e)}")
                    validation_results["zk_proof_validation"] = False
                    validation_results["error_conditions"].append("zk_proof_error")

            # Test fusion reproducibility
            if samples:
                sample = samples[0]
                if sample.has_features:
                    from .fusion import test_fusion_reproducibility

                    repro_results = test_fusion_reproducibility(
                        sample.fingerprint_features,
                        sample.face_features,
                        n_iterations=5,
                    )

                    validation_results["fusion_reproducibility"] = repro_results[
                        "is_reproducible"
                    ]
                    if not repro_results["is_reproducible"]:
                        validation_results["error_conditions"].append(
                            "fusion_non_reproducible"
                        )

            # Generate summary
            validation_results["validation_summary"] = {
                "all_tests_passed": len(validation_results["error_conditions"]) == 0,
                "tests_run": 3,
                "tests_passed": sum(
                    [
                        validation_results["template_consistency"],
                        validation_results["zk_proof_validation"],
                        validation_results["fusion_reproducibility"],
                    ]
                ),
                "error_count": len(validation_results["error_conditions"]),
            }

            logger.info(
                "System validation completed",
                all_tests_passed=validation_results["validation_summary"][
                    "all_tests_passed"
                ],
                error_count=validation_results["validation_summary"]["error_count"],
            )

            return validation_results

        except Exception as e:
            raise TestExecutionError(
                f"System validation failed: {str(e)}",
                test_type="validation",
                test_stage="execution",
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute the complete test suite including all test categories.

        This is the main entry point for comprehensive system testing.
        It orchestrates all test phases and generates complete results
        suitable for academic publication.

        Returns
        -------
        Dict[str, Any]
            Comprehensive test results including all test categories.

        Raises
        ------
        TestExecutionError
            If any critical test phase fails.
        """
        logger.info(
            "Starting comprehensive test suite execution",
            test_session_id=self.test_session_id,
        )

        test_start_time = time.time()

        try:
            # Phase 1: Load and prepare samples
            logger.info("Phase 1: Loading and preparing biometric samples")
            samples = self._load_and_prepare_samples()

            # Phase 2: Extract features
            logger.info("Phase 2: Extracting biometric features")
            samples = self._extract_biometric_features(samples)

            # Phase 3: Generate templates
            logger.info("Phase 3: Generating biometric templates")
            samples = self._generate_biometric_templates(samples)

            self.processed_samples = samples

            # Phase 4: Quality assessment
            logger.info("Phase 4: Running quality assessment")
            quality_results = self.run_quality_assessment(samples)

            # Phase 5: FMR/FNMR testing
            logger.info("Phase 5: Running FMR/FNMR tests")
            fmr_fnmr_results = self.run_fmr_fnmr_test(samples)

            # Phase 6: Performance benchmarking
            logger.info("Phase 6: Running performance benchmarks")
            performance_results = self.run_performance_test(samples)

            # Phase 7: System validation
            logger.info("Phase 7: Running system validation")
            validation_results = self.run_system_validation(samples)

            # Compile comprehensive results
            total_time = time.time() - test_start_time

            comprehensive_results = {
                "test_session_id": self.test_session_id,
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_execution_time_seconds": total_time,
                "dataset_statistics": (
                    self.dataset_stats.to_dict() if self.dataset_stats else {}
                ),
                "sample_processing": {
                    "total_samples_loaded": len(samples),
                    "successfully_processed": len(self.processed_samples),
                    "processing_success_rate": (
                        len(self.processed_samples) / len(samples) if samples else 0
                    ),
                },
                "quality_assessment": quality_results,
                "fmr_fnmr_analysis": fmr_fnmr_results,
                "performance_benchmarks": performance_results,
                "system_validation": validation_results,
                "test_configuration": self.test_config,
                "system_configuration": {
                    "enable_zk_proofs": self.enable_zk_proofs,
                    "enable_quality_checks": self.enable_quality_checks,
                },
                "individual_test_results": [
                    result.to_dict() for result in self.test_results
                ],
            }

            # Log results to disk
            logger.info("Saving test results to disk")
            result_file = self.result_logger.log_test_results(
                self.test_results, comprehensive_results
            )

            comprehensive_results["results_file"] = str(result_file)

            logger.info(
                "Comprehensive test suite completed successfully",
                total_time_seconds=total_time,
                total_tests=len(self.test_results),
                results_file=str(result_file),
            )

            return comprehensive_results

        except Exception as e:
            total_time = time.time() - test_start_time

            logger.error(
                "Test suite execution failed",
                error=str(e),
                error_type=type(e).__name__,
                execution_time_seconds=total_time,
                test_session_id=self.test_session_id,
            )

            if isinstance(e, TestExecutionError):
                raise
            else:
                raise TestExecutionError(
                    f"Unexpected error in test suite: {str(e)}",
                    test_type="comprehensive",
                    test_stage="execution",
                )
