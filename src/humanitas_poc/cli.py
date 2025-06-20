import sys
import argparse
from typing import Optional, List
import structlog

from .test_harness import TestHarness
from .dataset_loader import DatasetLoader
from .config import PROCESSED_DATA_DIR, RESULTS_PATH
from .exceptions import HumanitasPocError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class HumanitasCLI:
    """Main command-line interface for the HUMANITAS POC system."""

    def __init__(self) -> None:
        self.parser = self._create_argument_parser()
        logger.info("HUMANITAS POC CLI initialized")

    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Creates the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            prog="humanitas-poc",
            description="HUMANITAS POC - Biometric Uniqueness Validation System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", required=True)
        self._add_test_command(subparsers)

        return parser

    def _add_test_command(self, subparsers) -> None:
        """Add the 'test' command and its arguments."""
        test_parser = subparsers.add_parser(
            "test",
            help="Run comprehensive biometric system tests on the pre-processed dataset.",
        )

        test_parser.add_argument(
            "--max-samples",
            type=int,
            default=0,  # Default to all samples in the processed directory
            help="Maximum number of samples to test (0 for all). Default: 0.",
        )
        test_parser.add_argument(
            "--disable-zk-proofs",
            action="store_true",
            help="Disable ZK proof generation and testing.",
        )

    def _execute_test_command(self, args: argparse.Namespace) -> int:
        """
        Execute the test command using the pre-processed dataset.
        """
        logger.info("Starting comprehensive biometric system testing...")

        try:
            # --- THE FIX IS HERE ---
            # We now ONLY use the clean, pre-processed data directory.

            if not PROCESSED_DATA_DIR.is_dir() or not any(PROCESSED_DATA_DIR.iterdir()):
                logger.error(
                    "Processed data directory is empty or does not exist.",
                    path=str(PROCESSED_DATA_DIR),
                )
                print("\n[FATAL ERROR] Clean dataset not found.", file=sys.stderr)
                print(
                    "Please ensure you have run the 'scripts/preprocess_datasets.py' script successfully.",
                    file=sys.stderr,
                )
                return 1

            # Initialize the loader to point ONLY to the clean data
            max_samples = args.max_samples if args.max_samples > 0 else None
            dataset_loader = DatasetLoader(
                processed_dir=PROCESSED_DATA_DIR, max_samples=max_samples
            )

            # Initialize and run the test harness
            test_harness = TestHarness(
                dataset_loader=dataset_loader,
                output_path=RESULTS_PATH,
                enable_zk_proofs=not args.disable_zk_proofs,
            )

            results = test_harness.run_all_tests()
            self._display_test_summary(results)

            logger.info("Test execution completed successfully.")
            return 0

        except HumanitasPocError as e:
            logger.error(f"A known application error occurred: {e}", exc_info=True)
            print(f"\n[ERROR] {e}", file=sys.stderr)
            return 1
        except Exception as e:
            logger.error(f"An unexpected fatal error occurred: {e}", exc_info=True)
            print(
                f"\n[FATAL ERROR] An unexpected error occurred during testing: {e}",
                file=sys.stderr,
            )
            return 1

    def _display_test_summary(self, results: dict) -> None:
        """Display a summary of the test results."""
        print("\n" + "=" * 80)
        print("HUMANITAS POC - TEST RESULTS SUMMARY")
        print("=" * 80)
        # Add a simplified summary display here in the future if needed
        print(f"Test Session ID: {results.get('test_session_id', 'N/A')}")
        print(
            f"Total Execution Time: {results.get('total_execution_time_seconds', 0):.2f} seconds"
        )
        fmr_fnmr = results.get("fmr_fnmr_analysis", {})
        print(f"  False Match Rate (FMR): {fmr_fnmr.get('fmr', 'N/A')}")
        print(f"  False Non-Match Rate (FNMR): {fmr_fnmr.get('fnmr', 'N/A')}")
        print(f"Results saved to directory: {RESULTS_PATH}")
        print("=" * 80)

    def run_from_args(self, args_list: Optional[List[str]] = None) -> int:
        """Run the CLI with provided arguments."""
        try:
            args = self.parser.parse_args(args_list)
            if args.command == "test":
                return self._execute_test_command(args)
            else:
                self.parser.print_help()
                return 1
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.", file=sys.stderr)
            return 130


def main() -> int:
    """Main entry point for the CLI."""
    cli = HumanitasCLI()
    return cli.run_from_args()


if __name__ == "__main__":
    sys.exit(main())
