"""
Performance benchmarking for the HUMANITAS POC system.

This module provides comprehensive performance benchmarking for all components
of the biometric processing pipeline. It measures execution times, memory usage,
and throughput for academic performance analysis and system optimization.

The benchmarks generate detailed metrics suitable for academic publications
and comparative analysis with other biometric systems.
"""

import time
import psutil
import gc
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import structlog

from .data_models import BiometricSample
from .feature_extraction import extract_fingerprint_features, extract_facial_features
from .fusion import BiometricFusionEngine, fuse_features
from .template_generator import SecureTemplateGenerator
from .zk_prover import ZkProver
from .quality_assessment import BiometricQualityAssessor
from .exceptions import PerformanceBenchmarkError
from .constants import BENCHMARK_ITERATIONS

# Initialize structured logger
logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """
    Comprehensive benchmark result for a single component.

    Attributes
    ----------
    component_name : str
        Name of the benchmarked component.
    avg_time_ms : float
        Average execution time in milliseconds.
    std_time_ms : float
        Standard deviation of execution times.
    min_time_ms : float
        Minimum execution time.
    max_time_ms : float
        Maximum execution time.
    median_time_ms : float
        Median execution time.
    throughput_ops_per_sec : float
        Operations per second.
    memory_usage_mb : float
        Peak memory usage in MB.
    iterations : int
        Number of benchmark iterations.
    """

    component_name: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    iterations: int


class MemoryProfiler:
    """
    Simple memory profiler for tracking memory usage during operations.

    Examples
    --------
    >>> with MemoryProfiler() as profiler:
    ...     # Perform memory-intensive operation
    ...     result = some_operation()
    >>> print(f"Peak memory: {profiler.peak_memory_mb} MB")
    """

    def __init__(self) -> None:
        self.initial_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.process = psutil.Process()

    def __enter__(self) -> "MemoryProfiler":
        gc.collect()  # Clean up before measurement
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = self.initial_memory_mb
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        final_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, final_memory_mb)

    def update_peak(self) -> None:
        """Update peak memory usage with current memory."""
        current_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarker for biometric system components.

    This class provides detailed performance analysis including execution time,
    memory usage, and throughput measurements for all major components of the
    biometric processing pipeline.

    Parameters
    ----------
    default_iterations : int, default=BENCHMARK_ITERATIONS
        Default number of iterations for benchmarking.
    enable_memory_profiling : bool, default=True
        Whether to profile memory usage during benchmarks.
    warmup_iterations : int, default=5
        Number of warmup iterations before measurement.

    Examples
    --------
    >>> benchmarker = PerformanceBenchmarker()
    >>> sample = load_sample()
    >>> results = benchmarker.benchmark_all_components(sample)
    >>> print(f"Total time: {results['total_time_ms']} ms")
    """

    def __init__(
        self,
        default_iterations: int = BENCHMARK_ITERATIONS,
        enable_memory_profiling: bool = True,
        warmup_iterations: int = 5,
    ) -> None:
        self.default_iterations = default_iterations
        self.enable_memory_profiling = enable_memory_profiling
        self.warmup_iterations = warmup_iterations

        # Initialize components for benchmarking
        self.fusion_engine = BiometricFusionEngine()
        self.template_generator = SecureTemplateGenerator()
        self.zk_prover = ZkProver()
        self.quality_assessor = BiometricQualityAssessor()

        logger.info(
            "PerformanceBenchmarker initialized",
            default_iterations=default_iterations,
            enable_memory_profiling=enable_memory_profiling,
            warmup_iterations=warmup_iterations,
        )

    def _benchmark_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = None,
        component_name: str = "unknown",
    ) -> BenchmarkResult:
        """
        Benchmark a single function with comprehensive metrics.

        Parameters
        ----------
        func : Callable
            Function to benchmark.
        args : tuple, default=()
            Positional arguments for the function.
        kwargs : dict, default=None
            Keyword arguments for the function.
        iterations : int, default=None
            Number of iterations. Uses default if None.
        component_name : str, default="unknown"
            Name of the component being benchmarked.

        Returns
        -------
        BenchmarkResult
            Comprehensive benchmark results.
        """
        if kwargs is None:
            kwargs = {}
        if iterations is None:
            iterations = self.default_iterations

        logger.debug(
            f"Starting benchmark for {component_name}",
            iterations=iterations,
            function_name=func.__name__,
        )

        # Warmup iterations
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Actual benchmark iterations
        execution_times = []
        memory_usage_mb = 0.0

        for i in range(iterations):
            gc.collect()  # Clean memory before each iteration

            if self.enable_memory_profiling:
                with MemoryProfiler() as profiler:
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        end_time = time.perf_counter()
                        execution_time = (end_time - start_time) * 1000  # Convert to ms
                        execution_times.append(execution_time)
                    except Exception as e:
                        logger.warning(
                            f"Benchmark iteration {i} failed for {component_name}",
                            error=str(e),
                        )
                        continue

                memory_usage_mb = max(memory_usage_mb, profiler.peak_memory_mb)
            else:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    execution_time = (end_time - start_time) * 1000
                    execution_times.append(execution_time)
                except Exception as e:
                    logger.warning(
                        f"Benchmark iteration {i} failed for {component_name}",
                        error=str(e),
                    )
                    continue

        if not execution_times:
            raise PerformanceBenchmarkError(
                f"All benchmark iterations failed for {component_name}",
                benchmark_type=component_name,
            )

        # Calculate statistics
        execution_times_array = np.array(execution_times)
        avg_time_ms = float(np.mean(execution_times_array))
        std_time_ms = float(np.std(execution_times_array))
        min_time_ms = float(np.min(execution_times_array))
        max_time_ms = float(np.max(execution_times_array))
        median_time_ms = float(np.median(execution_times_array))

        # Calculate throughput (operations per second)
        throughput_ops_per_sec = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

        result = BenchmarkResult(
            component_name=component_name,
            avg_time_ms=avg_time_ms,
            std_time_ms=std_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            median_time_ms=median_time_ms,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            iterations=len(execution_times),
        )

        logger.info(
            f"Benchmark completed for {component_name}",
            avg_time_ms=avg_time_ms,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            successful_iterations=len(execution_times),
        )

        return result

    def benchmark_feature_extraction(
        self, sample: BiometricSample, iterations: int = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark biometric feature extraction components.

        Parameters
        ----------
        sample : BiometricSample
            Sample to use for benchmarking.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        Dict[str, BenchmarkResult]
            Benchmark results for each extraction component.
        """
        logger.info("Benchmarking feature extraction components")

        results = {}

        # Benchmark fingerprint feature extraction
        if sample.fingerprint_paths:
            fp_path = sample.fingerprint_paths[0]  # Use first fingerprint
            results["fingerprint_extraction"] = self._benchmark_function(
                extract_fingerprint_features,
                args=(fp_path,),
                iterations=iterations,
                component_name="fingerprint_extraction",
            )

        # Benchmark face feature extraction
        if sample.face_path:
            results["face_extraction"] = self._benchmark_function(
                extract_facial_features,
                args=(sample.face_path,),
                iterations=iterations,
                component_name="face_extraction",
            )

        return results

    def benchmark_quality_assessment(
        self, sample: BiometricSample, iterations: int = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark quality assessment components.

        Parameters
        ----------
        sample : BiometricSample
            Sample to use for benchmarking.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        Dict[str, BenchmarkResult]
            Benchmark results for quality assessment.
        """
        logger.info("Benchmarking quality assessment components")

        results = {}

        # Benchmark face quality assessment
        if sample.face_path:
            results["face_quality"] = self._benchmark_function(
                self.quality_assessor.assess_face_quality,
                args=(sample.face_path,),
                iterations=iterations,
                component_name="face_quality_assessment",
            )

        # Benchmark fingerprint quality assessment
        if sample.fingerprint_paths:
            fp_path = sample.fingerprint_paths[0]
            results["fingerprint_quality"] = self._benchmark_function(
                self.quality_assessor.assess_fingerprint_quality,
                args=(fp_path,),
                iterations=iterations,
                component_name="fingerprint_quality_assessment",
            )

        return results

    def benchmark_fusion(
        self, sample: BiometricSample, iterations: int = None
    ) -> BenchmarkResult:
        """
        Benchmark multimodal feature fusion.

        Parameters
        ----------
        sample : BiometricSample
            Sample with extracted features for benchmarking.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        BenchmarkResult
            Benchmark results for fusion.
        """
        logger.info("Benchmarking feature fusion")

        if not sample.has_features:
            raise PerformanceBenchmarkError(
                "Sample must have extracted features for fusion benchmarking",
                benchmark_type="fusion",
            )

        # Prepare fingerprint features list
        fp_features_list = [
            sample.fingerprint_features[i]
            for i in range(sample.fingerprint_features.shape[0])
        ]

        return self._benchmark_function(
            fuse_features,
            args=(fp_features_list, sample.face_features),
            iterations=iterations,
            component_name="feature_fusion",
        )

    def benchmark_template_generation(
        self, fused_vector: np.ndarray, iterations: int = None
    ) -> BenchmarkResult:
        """
        Benchmark secure template generation.

        Parameters
        ----------
        fused_vector : np.ndarray
            Fused feature vector for benchmarking.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        BenchmarkResult
            Benchmark results for template generation.
        """
        logger.info("Benchmarking template generation")

        return self._benchmark_function(
            self.template_generator.generate_secure_template,
            args=(fused_vector,),
            iterations=iterations,
            component_name="template_generation",
        )

    def benchmark_zk_proof_generation(
        self, sample: BiometricSample, iterations: int = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark zero-knowledge proof generation and verification.

        Parameters
        ----------
        sample : BiometricSample
            Sample with template for ZK proof benchmarking.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        Dict[str, BenchmarkResult]
            Benchmark results for ZK proof operations.
        """
        logger.info("Benchmarking ZK proof operations")

        if not sample.has_template:
            raise PerformanceBenchmarkError(
                "Sample must have template for ZK proof benchmarking",
                benchmark_type="zk_proof",
            )

        results = {}

        # Benchmark setup (only once, as it's expensive)
        setup_result = self._benchmark_function(
            self.zk_prover.setup,
            iterations=1,  # Setup is expensive, only do once
            component_name="zk_setup",
        )
        results["zk_setup"] = setup_result

        # Get keys for proof generation/verification
        proving_key, verifying_key = self.zk_prover.setup()

        # Prepare private inputs
        fused_vector = np.frombuffer(sample.fused_template, dtype=np.float32)
        private_inputs = {
            "fused_vector": fused_vector,
            "salt": sample.salt,
            "template_hash": sample.template_hash,
        }

        # Benchmark proof generation
        results["zk_prove"] = self._benchmark_function(
            self.zk_prover.prove,
            args=(proving_key, private_inputs),
            iterations=min(
                iterations or self.default_iterations, 10
            ),  # Limit expensive operations
            component_name="zk_proof_generation",
        )

        # Generate a proof for verification benchmarking
        proof = self.zk_prover.prove(proving_key, private_inputs)
        public_inputs = {"template_hash": sample.template_hash}

        # Benchmark proof verification
        results["zk_verify"] = self._benchmark_function(
            self.zk_prover.verify,
            args=(verifying_key, proof, public_inputs),
            iterations=iterations,
            component_name="zk_proof_verification",
        )

        return results

    def benchmark_end_to_end_pipeline(
        self, sample: BiometricSample, iterations: int = None
    ) -> BenchmarkResult:
        """
        Benchmark the complete end-to-end biometric processing pipeline.

        Parameters
        ----------
        sample : BiometricSample
            Sample to process through the complete pipeline.
        iterations : int, default=None
            Number of benchmark iterations.

        Returns
        -------
        BenchmarkResult
            Benchmark results for the complete pipeline.
        """
        logger.info("Benchmarking end-to-end pipeline")

        def complete_pipeline():
            """Complete biometric processing pipeline."""
            # Feature extraction
            fp_features = []
            for fp_path in sample.fingerprint_paths:
                features = extract_fingerprint_features(fp_path)
                fp_features.append(features)

            face_features = extract_facial_features(sample.face_path)

            # Feature fusion
            fused_vector = fuse_features(fp_features, face_features)

            # Template generation
            template_hash, salt = self.template_generator.generate_secure_template(
                fused_vector
            )

            return template_hash, salt

        return self._benchmark_function(
            complete_pipeline,
            iterations=min(
                iterations or self.default_iterations, 20
            ),  # Limit for expensive operation
            component_name="end_to_end_pipeline",
        )

    def benchmark_all_components(
        self,
        sample: Optional[BiometricSample] = None,
        iterations: int = None,
        enable_zk_proofs: bool = True,
        include_memory_profiling: bool = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks on all system components.

        This is the main entry point for performance benchmarking. It tests
        all major components and provides detailed performance analysis.

        Parameters
        ----------
        sample : Optional[BiometricSample], default=None
            Sample to use for benchmarking. If None, creates synthetic data.
        iterations : int, default=None
            Number of iterations for each benchmark.
        enable_zk_proofs : bool, default=True
            Whether to include ZK proof benchmarking.
        include_memory_profiling : bool, default=None
            Whether to profile memory usage. Uses class default if None.

        Returns
        -------
        Dict[str, Any]
            Comprehensive benchmark results for all components.

        Raises
        ------
        PerformanceBenchmarkError
            If benchmarking fails critically.
        """
        logger.info("Starting comprehensive performance benchmarking")

        if include_memory_profiling is not None:
            original_memory_profiling = self.enable_memory_profiling
            self.enable_memory_profiling = include_memory_profiling

        try:
            benchmark_start_time = time.time()
            all_results = {}

            # Use provided sample or create synthetic one for benchmarking
            if sample is None:
                # Create synthetic sample for benchmarking
                sample = self._create_synthetic_sample()

            # Ensure sample has features for fusion/template benchmarks
            if not sample.has_features:
                sample = self._prepare_sample_features(sample)

            # Benchmark feature extraction
            try:
                extraction_results = self.benchmark_feature_extraction(
                    sample, iterations
                )
                all_results.update(extraction_results)
            except Exception as e:
                logger.warning(f"Feature extraction benchmarking failed: {str(e)}")

            # Benchmark quality assessment
            try:
                quality_results = self.benchmark_quality_assessment(sample, iterations)
                all_results.update(quality_results)
            except Exception as e:
                logger.warning(f"Quality assessment benchmarking failed: {str(e)}")

            # Benchmark fusion
            try:
                fusion_result = self.benchmark_fusion(sample, iterations)
                all_results["fusion"] = fusion_result
            except Exception as e:
                logger.warning(f"Fusion benchmarking failed: {str(e)}")

            # Benchmark template generation
            try:
                if sample.has_features:
                    fp_features_list = [
                        sample.fingerprint_features[i]
                        for i in range(sample.fingerprint_features.shape[0])
                    ]
                    fused_vector = fuse_features(fp_features_list, sample.face_features)

                    template_result = self.benchmark_template_generation(
                        fused_vector, iterations
                    )
                    all_results["template_generation"] = template_result
            except Exception as e:
                logger.warning(f"Template generation benchmarking failed: {str(e)}")

            # Benchmark ZK proofs if enabled
            if enable_zk_proofs:
                try:
                    # Ensure sample has template
                    if not sample.has_template:
                        sample = self._prepare_sample_template(sample)

                    zk_results = self.benchmark_zk_proof_generation(sample, iterations)
                    all_results.update(zk_results)
                except Exception as e:
                    logger.warning(f"ZK proof benchmarking failed: {str(e)}")

            # Benchmark end-to-end pipeline
            try:
                e2e_result = self.benchmark_end_to_end_pipeline(sample, iterations)
                all_results["end_to_end"] = e2e_result
            except Exception as e:
                logger.warning(f"End-to-end benchmarking failed: {str(e)}")

            # Calculate summary statistics
            total_benchmark_time = time.time() - benchmark_start_time

            # Extract timing information
            component_times = {}
            total_avg_time = 0.0

            for component_name, result in all_results.items():
                if isinstance(result, BenchmarkResult):
                    component_times[component_name] = int(result.avg_time_ms)
                    total_avg_time += result.avg_time_ms

            summary_results = {
                "component_results": {
                    name: {
                        "avg_time_ms": result.avg_time_ms,
                        "throughput_ops_per_sec": result.throughput_ops_per_sec,
                        "memory_usage_mb": result.memory_usage_mb,
                        "iterations": result.iterations,
                    }
                    for name, result in all_results.items()
                    if isinstance(result, BenchmarkResult)
                },
                "component_times": component_times,
                "total_time_ms": int(total_avg_time),
                "total_benchmark_time_seconds": total_benchmark_time,
                "benchmark_metadata": {
                    "iterations_per_component": iterations or self.default_iterations,
                    "memory_profiling_enabled": self.enable_memory_profiling,
                    "zk_proofs_enabled": enable_zk_proofs,
                    "components_tested": len(all_results),
                },
            }

            logger.info(
                "Comprehensive benchmarking completed",
                total_time_ms=summary_results["total_time_ms"],
                components_tested=len(all_results),
                benchmark_time_seconds=total_benchmark_time,
            )

            return summary_results

        except Exception as e:
            if isinstance(e, PerformanceBenchmarkError):
                raise
            else:
                raise PerformanceBenchmarkError(
                    f"Comprehensive benchmarking failed: {str(e)}",
                    benchmark_type="comprehensive",
                )
        finally:
            # Restore original memory profiling setting
            if include_memory_profiling is not None:
                self.enable_memory_profiling = original_memory_profiling

    def _create_synthetic_sample(self) -> BiometricSample:
        """Create a synthetic biometric sample for benchmarking."""
        return BiometricSample(
            person_id="benchmark_synthetic",
            fingerprint_paths=["synthetic_fp1.png", "synthetic_fp2.png"],
            face_path="synthetic_face.jpg",
        )

    def _prepare_sample_features(self, sample: BiometricSample) -> BiometricSample:
        """Prepare synthetic features for a sample."""
        # Create synthetic features for benchmarking
        sample.fingerprint_features = np.random.randn(10, 512).astype(np.float32)
        sample.face_features = np.random.randn(128).astype(np.float32)
        return sample

    def _prepare_sample_template(self, sample: BiometricSample) -> BiometricSample:
        """Prepare synthetic template for a sample."""
        if not sample.has_features:
            sample = self._prepare_sample_features(sample)

        # Generate template from features
        fp_features_list = [
            sample.fingerprint_features[i]
            for i in range(sample.fingerprint_features.shape[0])
        ]
        fused_vector = fuse_features(fp_features_list, sample.face_features)
        template_hash, salt = self.template_generator.generate_secure_template(
            fused_vector
        )

        sample.fused_template = fused_vector.tobytes()
        sample.template_hash = template_hash
        sample.salt = salt

        return sample
