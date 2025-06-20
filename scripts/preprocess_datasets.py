"""
Dataset Pre-processing and Standardization Script for HUMANITAS POC.

This script reads a manually completed metadata mapping file to build a clean,
standardized, and ready-to-use dataset for the main application. It performs
the following steps:
1. Reads the `metadata_mapping.csv` file.
2. Filters for rows where both NIST and LFW IDs are mapped.
3. Creates a standardized directory structure in `data/processed/`.
4. Copies and renames fingerprint and face images to a consistent format.
5. Generates a detailed processing report.

Author: Generated for Humanitas Project
Date: 2025-06-19
"""

import csv
import re
import shutil
from pathlib import Path
import argparse
import sys


class DatasetPreprocessor:
    """Orchestrates the data cleaning and standardization pipeline."""

    def __init__(self, base_dir=".", mapping_file="metadata_mapping.csv"):
        """Initialize the preprocessor."""
        self.base_dir = Path(base_dir).resolve()
        self.raw_data_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed" / "humanitas_standard"
        self.mapping_file = self.base_dir / mapping_file

        self.nist_dir = self.raw_data_dir / "nist_sd302"
        self.lfw_dir = self.raw_data_dir / "lfw"

        self.report = []
        self.error_count = 0
        self.processed_count = 0

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def log(self, message, level="INFO"):
        """Simple logging to console and report."""
        print(f"[{level}] {message}")
        if level in ["ERROR", "WARNING", "INFO"]:
            self.report.append(f"[{level}] {message}")

    def validate_setup(self):
        """Validate that all required paths and files exist."""
        self.log("Validating setup...")
        if not self.mapping_file.exists():
            self.log(f"Mapping file not found at: {self.mapping_file}", "ERROR")
            raise FileNotFoundError(
                "Mapping file not found. Please create it from the template and fill it out."
            )
        if not self.nist_dir.is_dir():
            self.log(
                f"NIST directory not found or is not a directory: {self.nist_dir}",
                "ERROR",
            )
            raise FileNotFoundError(f"NIST directory not found: {self.nist_dir}")
        if not self.lfw_dir.is_dir():
            self.log(
                f"LFW directory not found or is not a directory: {self.lfw_dir}",
                "ERROR",
            )
            raise FileNotFoundError(f"LFW directory not found: {self.lfw_dir}")

        # Create the main output directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.log("Setup validation successful.")
        return True

    def read_mapping_file(self):
        """Read and parse the manually completed mapping CSV."""
        self.log(f"Reading mapping file: {self.mapping_file}")
        valid_mappings = []
        with open(self.mapping_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nist_id = row.get("nist_id", "").strip()
                lfw_id = row.get("lfw_id", "").strip()
                humanitas_id = row.get("humanitas_id", "").strip()

                # Process only rows that have been manually mapped
                if nist_id and lfw_id and humanitas_id:
                    valid_mappings.append(row)

        if not valid_mappings:
            self.log(
                "No valid, completed mappings found in the CSV. Please fill in the 'lfw_id' column.",
                "ERROR",
            )
        else:
            self.log(f"Found {len(valid_mappings)} valid mappings to process.")
        return valid_mappings

    def find_nist_files(self, nist_id):
        """
        Find all fingerprint files for a given NIST ID.
        This version automatically pads the ID with leading zeros to match
        common filename formats (e.g., '2651' becomes '00002651').
        """
        found_files = []
        # Regex to match filenames starting with the nist_id
        padded_nist_id = nist_id.zfill(8)
        nist_id_pattern = re.compile(f"^{re.escape(padded_nist_id)}")

        for filepath in self.nist_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in self.image_extensions:
                if nist_id_pattern.search(filepath.stem):
                    found_files.append(filepath)
        return sorted(found_files)

    def find_lfw_file(self, lfw_id):
        """Find the primary face image file for a given LFW ID from a nested directory."""
        person_dir = self.lfw_dir / lfw_id
        if not person_dir.is_dir():
            self.log(f"LFW person directory not found: {person_dir}", "WARNING")
            return None

        image_files = sorted(
            [
                p
                for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.image_extensions
            ]
        )

        if not image_files:
            self.log(f"No images found in LFW directory: {person_dir}", "WARNING")
            return None

        # Heuristic: return the '..._0001.jpg' file if it exists, otherwise the first file.
        for img_file in image_files:
            if img_file.stem.endswith("_0001"):
                return img_file
        return image_files[0]

    def process_row(self, row):
        """Process a single row from the mapping file."""
        humanitas_id = row["humanitas_id"]
        nist_id = row["nist_id"]
        lfw_id = row["lfw_id"]

        self.log(f"Processing {humanitas_id} (NIST: {nist_id}, LFW: {lfw_id})")

        # Create destination directory
        dest_dir = self.processed_dir / humanitas_id
        dest_dir.mkdir(exist_ok=True)

        # --- Process Fingerprints ---
        nist_files = self.find_nist_files(nist_id)
        if not nist_files:
            self.log(f"No fingerprint files found for NIST ID: {nist_id}", "ERROR")
            self.error_count += 1
            return

        # Take only the first 10 fingerprints to ensure consistency
        files_to_copy = nist_files[:10]

        fp_copied_count = 0
        for i, src_path in enumerate(files_to_copy):
            dest_name = f"fp_{i+1:02d}{src_path.suffix}"
            dest_path = dest_dir / dest_name
            shutil.copy2(src_path, dest_path)
            fp_copied_count += 1
        self.log(f"  Copied {fp_copied_count} fingerprint images.", "DEBUG")

        # --- Process Face ---
        lfw_file = self.find_lfw_file(lfw_id)
        if not lfw_file:
            self.log(f"No face image found for LFW ID: {lfw_id}", "ERROR")
            self.error_count += 1
            return

        dest_name = f"face{lfw_file.suffix}"
        dest_path = dest_dir / dest_name
        shutil.copy2(lfw_file, dest_path)
        self.log("  Copied 1 face image.", "DEBUG")

        self.processed_count += 1
        self.report.append(
            f"[SUCCESS] {humanitas_id}: Processed {fp_copied_count} fingerprints and 1 face."
        )

    def write_report(self):
        """Write the final processing report to a text file."""
        report_file = self.base_dir / "preprocessing_report.txt"
        self.log(f"Writing processing report to: {report_file}")

        summary = [
            "HUMANITAS Dataset Pre-processing Report",
            "=========================================",
            f"Date: {__import__('datetime').datetime.now().isoformat()}",
            f"Mapping File: {self.mapping_file}",
            f"Output Directory: {self.processed_dir}",
            "",
            f"Total People Processed: {self.processed_count}",
            f"Total Errors/Missing Files: {self.error_count}",
            "=========================================\n",
        ]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
            f.write("\nDetailed Log:\n")
            f.write("-------------\n")
            f.write("\n".join(self.report))

        self.log("Report generation complete.")

    def run(self):
        """Execute the full pre-processing pipeline."""
        try:
            self.validate_setup()

            mappings = self.read_mapping_file()
            if not mappings:
                return  # Error message already logged in read_mapping_file

            for row in mappings:
                self.process_row(row)

            self.write_report()

            print("\n=========================================")
            print("PRE-PROCESSING COMPLETE")
            print(f"Successfully processed {self.processed_count} individuals.")
            print(f"Encountered {self.error_count} errors.")
            print("See full details in preprocessing_report.txt")
            print("=========================================")
            print(f"Your clean dataset is ready at: {self.processed_dir}")

        except Exception as e:
            self.log(f"A critical error occurred: {e}", "ERROR")
            self.write_report()  # Write report even on crash
            raise


def main():
    """Main function to run the script via command line."""
    parser = argparse.ArgumentParser(
        description="Pre-process and standardize HUMANITAS biometric datasets."
    )
    parser.add_argument(
        "--mapping-file",
        default="metadata_mapping.csv",
        help="Name of the mapping CSV file in the project root (default: metadata_mapping.csv)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory of the HUMANITAS POC project (default: current directory)",
    )

    args = parser.parse_args()

    try:
        preprocessor = DatasetPreprocessor(
            base_dir=args.base_dir, mapping_file=args.mapping_file
        )
        preprocessor.run()
        sys.exit(0)
    except FileNotFoundError as e:
        print(
            f"\n[FATAL ERROR] A required file or directory was not found: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        # For more detailed error info, uncomment the following line
        # import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
