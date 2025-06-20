"""
Utility functions and decorators for the HUMANITAS POC system.

This module provides general-purpose helper functions, decorators, and utilities
that are used across the biometric system. It includes timing decorators,
ID generation, validation helpers, and other common functionality.

All utilities are designed to support the academic research requirements
with comprehensive logging and error handling.
"""

import time
import uuid
import hashlib
import functools
import inspect
from typing import Any, Callable, Dict, List, TypeVar, Union
from pathlib import Path
import structlog

# Initialize structured logger
logger = structlog.get_logger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """
    Decorator to measure and log function execution time.

    This decorator automatically measures the execution time of functions
    and logs the results using structured logging. It preserves function
    metadata and supports both synchronous and asynchronous functions.

    Parameters
    ----------
    func : Callable
        Function to be timed.

    Returns
    -------
    Callable
        Wrapped function with timing capability.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    >>> result = slow_function()  # Logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.debug(
                "Function execution completed",
                function_name=func.__name__,
                module=func.__module__,
                execution_time_ms=execution_time,
                success=True,
            )

            return result

        except Exception as e:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            logger.error(
                "Function execution failed",
                function_name=func.__name__,
                module=func.__module__,
                execution_time_ms=execution_time,
                error=str(e),
                error_type=type(e).__name__,
                success=False,
            )

            raise

    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to retry function execution on failure.

    This decorator automatically retries function execution when specific
    exceptions occur, with configurable delay and backoff strategies.
    Useful for handling transient failures in biometric processing.

    Parameters
    ----------
    max_attempts : int, default=3
        Maximum number of retry attempts.
    delay : float, default=1.0
        Initial delay between retries in seconds.
    backoff : float, default=2.0
        Backoff multiplier for delay.
    exceptions : tuple, default=(Exception,)
        Tuple of exception types to catch and retry.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @retry(max_attempts=3, delay=0.5)
    ... def unreliable_function():
    ...     # Function that might fail temporarily
    ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:  # Don't sleep on last attempt
                        logger.warning(
                            f"Function {func.__name__} failed, retrying",
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay_seconds=current_delay,
                            error=str(e),
                        )

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Function {func.__name__} failed after all retries",
                            total_attempts=max_attempts,
                            final_error=str(e),
                        )

            # If we get here, all attempts failed
            raise last_exception

        return wrapper

    return decorator


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str = "Input validation failed",
) -> Callable[[F], F]:
    """
    Decorator to validate function inputs.

    This decorator applies validation to function arguments before execution.
    Useful for ensuring biometric data meets quality requirements.

    Parameters
    ----------
    validation_func : Callable[[Any], bool]
        Function that returns True if input is valid.
    error_message : str, default="Input validation failed"
        Error message for validation failures.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> def is_positive(x):
    ...     return x > 0
    >>> @validate_input(is_positive, "Value must be positive")
    ... def process_value(x):
    ...     return x * 2
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, arg in enumerate(args):
                if not validation_func(arg):
                    raise ValueError(f"{error_message} (argument {i}): {arg}")

            # Validate keyword arguments
            for key, value in kwargs.items():
                if not validation_func(value):
                    raise ValueError(f"{error_message} (argument {key}): {value}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def generate_test_id(prefix: str = "test") -> str:
    """
    Generate a unique test identifier.

    Creates a unique identifier suitable for tracking test executions
    and results across the system. The ID includes a timestamp component
    for chronological ordering.

    Parameters
    ----------
    prefix : str, default="test"
        Prefix for the generated ID.

    Returns
    -------
    str
        Unique test identifier.

    Examples
    --------
    >>> test_id = generate_test_id("fmr")
    >>> print(test_id)  # e.g., "fmr_20240101_123456_abc123"
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_suffix = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_suffix}"


def generate_session_id() -> str:
    """
    Generate a unique session identifier.

    Returns
    -------
    str
        Unique session identifier.

    Examples
    --------
    >>> session_id = generate_session_id()
    >>> print(len(session_id))  # 32 characters
    """
    return str(uuid.uuid4()).replace("-", "")


def hash_data(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """
    Generate hash of data for integrity verification.

    This function creates cryptographic hashes of data for integrity
    checking and deduplication purposes.

    Parameters
    ----------
    data : Union[str, bytes]
        Data to hash.
    algorithm : str, default="sha256"
        Hashing algorithm to use.

    Returns
    -------
    str
        Hexadecimal hash string.

    Raises
    ------
    ValueError
        If algorithm is not supported.

    Examples
    --------
    >>> hash_value = hash_data("test data")
    >>> print(len(hash_value))  # 64 characters for SHA-256
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero denominator.

    Parameters
    ----------
    numerator : float
        Numerator value.
    denominator : float
        Denominator value.
    default : float, default=0.0
        Default value when denominator is zero.

    Returns
    -------
    float
        Division result or default value.

    Examples
    --------
    >>> result = safe_divide(10, 2)  # Returns 5.0
    >>> result = safe_divide(10, 0)  # Returns 0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_bytes(bytes_value: int) -> str:
    """
    Format byte count as human-readable string.

    Parameters
    ----------
    bytes_value : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string with appropriate unit.

    Examples
    --------
    >>> print(format_bytes(1024))  # "1.0 KB"
    >>> print(format_bytes(1048576))  # "1.0 MB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string.

    Examples
    --------
    >>> print(format_duration(65))  # "1m 5s"
    >>> print(format_duration(3661))  # "1h 1m 1s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    path : Union[str, Path]
        Directory path to ensure.

    Returns
    -------
    Path
        Path object for the directory.

    Examples
    --------
    >>> dir_path = ensure_directory("./results/test")
    >>> assert dir_path.exists()
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing file information.

    Examples
    --------
    >>> info = get_file_info("data.txt")
    >>> print(info['size_bytes'])
    """
    path_obj = Path(file_path)

    if not path_obj.exists():
        return {"exists": False, "path": str(path_obj)}

    stat = path_obj.stat()

    return {
        "exists": True,
        "path": str(path_obj),
        "absolute_path": str(path_obj.absolute()),
        "size_bytes": stat.st_size,
        "size_human": format_bytes(stat.st_size),
        "modified_timestamp": stat.st_mtime,
        "is_file": path_obj.is_file(),
        "is_directory": path_obj.is_dir(),
        "suffix": path_obj.suffix,
        "stem": path_obj.stem,
        "parent": str(path_obj.parent),
    }


def flatten_dict(
    nested_dict: Dict[str, Any], separator: str = ".", prefix: str = ""
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    nested_dict : Dict[str, Any]
        Dictionary to flatten.
    separator : str, default="."
        Separator for nested keys.
    prefix : str, default=""
        Prefix for keys.

    Returns
    -------
    Dict[str, Any]
        Flattened dictionary.

    Examples
    --------
    >>> nested = {"a": {"b": {"c": 1}}}
    >>> flat = flatten_dict(nested)
    >>> print(flat)  # {"a.b.c": 1}
    """
    items = []

    for key, value in nested_dict.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Parameters
    ----------
    input_list : List[Any]
        List to chunk.
    chunk_size : int
        Size of each chunk.

    Returns
    -------
    List[List[Any]]
        List of chunks.

    Examples
    --------
    >>> chunks = chunk_list([1, 2, 3, 4, 5], 2)
    >>> print(chunks)  # [[1, 2], [3, 4], [5]]
    """
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    Get detailed information about a function's signature.

    Parameters
    ----------
    func : Callable
        Function to inspect.

    Returns
    -------
    Dict[str, Any]
        Function signature information.

    Examples
    --------
    >>> def example_func(a: int, b: str = "default") -> str:
    ...     return f"{a}: {b}"
    >>> sig_info = get_function_signature(example_func)
    >>> print(sig_info['parameters'])
    """
    signature = inspect.signature(func)

    parameters = {}
    for name, param in signature.parameters.items():
        param_info = {
            "name": name,
            "kind": param.kind.name,
            "annotation": (
                str(param.annotation)
                if param.annotation != inspect.Parameter.empty
                else None
            ),
            "default": (
                param.default if param.default != inspect.Parameter.empty else None
            ),
        }
        parameters[name] = param_info

    return {
        "function_name": func.__name__,
        "module": func.__module__,
        "docstring": func.__doc__,
        "parameters": parameters,
        "return_annotation": (
            str(signature.return_annotation)
            if signature.return_annotation != inspect.Signature.empty
            else None
        ),
    }


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.

    This class provides progress tracking with logging for academic
    operations that may take significant time.

    Parameters
    ----------
    total_items : int
        Total number of items to process.
    description : str, default="Processing"
        Description of the operation.
    log_interval : int, default=10
        Percentage interval for progress logging.

    Examples
    --------
    >>> tracker = ProgressTracker(100, "Processing samples")
    >>> for i in range(100):
    ...     tracker.update(i + 1)
    ...     # Process item
    """

    def __init__(
        self, total_items: int, description: str = "Processing", log_interval: int = 10
    ) -> None:
        self.total_items = total_items
        self.description = description
        self.log_interval = log_interval
        self.current_item = 0
        self.start_time = time.time()
        self.last_logged_percentage = 0

        logger.info(
            f"Starting {description}",
            total_items=total_items,
            log_interval_percent=log_interval,
        )

    def update(self, current_item: int) -> None:
        """
        Update progress tracker.

        Parameters
        ----------
        current_item : int
            Current item number (1-based).
        """
        self.current_item = current_item
        percentage = (current_item / self.total_items) * 100

        # Log at specified intervals
        if percentage >= self.last_logged_percentage + self.log_interval:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time * (self.total_items / current_item)
            remaining_time = estimated_total_time - elapsed_time

            logger.info(
                f"{self.description} progress",
                current_item=current_item,
                total_items=self.total_items,
                percentage=f"{percentage:.1f}%",
                elapsed_time_seconds=elapsed_time,
                estimated_remaining_seconds=remaining_time,
            )

            self.last_logged_percentage = (
                int(percentage // self.log_interval) * self.log_interval
            )

    def complete(self) -> None:
        """Mark progress as complete and log final statistics."""
        total_time = time.time() - self.start_time

        logger.info(
            f"{self.description} completed",
            total_items=self.total_items,
            total_time_seconds=total_time,
            items_per_second=self.total_items / total_time if total_time > 0 else 0,
        )


class ConfigurationValidator:
    """
    Utility class for validating configuration parameters.

    This class provides comprehensive validation for system configuration
    to ensure academic reproducibility and prevent runtime errors.

    Examples
    --------
    >>> validator = ConfigurationValidator()
    >>> is_valid = validator.validate_config({
    ...     "max_samples": 1000,
    ...     "enable_zk_proofs": True
    ... })
    """

    def __init__(self) -> None:
        self.validation_rules = {
            "max_samples": self._validate_positive_int,
            "enable_zk_proofs": self._validate_boolean,
            "random_seed": self._validate_optional_int,
            "output_path": self._validate_path,
            "log_level": self._validate_log_level,
        }

    def _validate_positive_int(self, value: Any) -> bool:
        """Validate positive integer."""
        return isinstance(value, int) and value > 0

    def _validate_boolean(self, value: Any) -> bool:
        """Validate boolean value."""
        return isinstance(value, bool)

    def _validate_optional_int(self, value: Any) -> bool:
        """Validate optional integer."""
        return value is None or isinstance(value, int)

    def _validate_path(self, value: Any) -> bool:
        """Validate path string."""
        if isinstance(value, (str, Path)):
            try:
                Path(value)
                return True
            except Exception:
                return False
        return False

    def _validate_log_level(self, value: Any) -> bool:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        return isinstance(value, str) and value.upper() in valid_levels

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.

        Returns
        -------
        bool
            True if configuration is valid.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        errors = []

        for key, value in config.items():
            if key in self.validation_rules:
                validator = self.validation_rules[key]
                if not validator(value):
                    errors.append(f"Invalid value for {key}: {value}")

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error("Configuration validation failed", errors=errors)
            raise ValueError(error_message)

        logger.debug("Configuration validation passed", config_keys=list(config.keys()))
        return True
