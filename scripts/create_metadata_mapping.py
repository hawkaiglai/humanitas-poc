"""
Dataset Metadata and Mapping Script (v1.1)

This script scans the unarranged NIST SD-302d and the pre-arranged LFW datasets
to generate a foundational CSV file for linking individuals.

This version is specifically adapted to handle the flat directory structure of
the NIST dataset and the nested structure of the LFW dataset from Kaggle.

Author: Generated for Humanitas Project
Date: 2025-06-19
"""

import csv
import re
from pathlib import Path
from collections import defaultdict
import argparse


class DatasetScanner:
    """Scanner class for processing NIST and LFW datasets."""

    def __init__(self, base_dir="."):
        """Initialize scanner with base directory."""
        self.base_dir = Path(base_dir).resolve()
        self.nist_dir = self.base_dir / "data" / "raw" / "nist_sd302"
        self.lfw_dir = self.base_dir / "data" / "raw" / "lfw"
        self.output_file = self.base_dir / "metadata_mapping_template.csv"

        self.nist_data = defaultdict(list)  # {person_id: [filepath, ...]}
        self.lfw_data = defaultdict(list)  # {person_id: [filepath, ...]}

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def extract_nist_id(self, filepath: Path) -> str | None:
        """
        Extracts the person ID from NIST SD-302d filenames.
        The format is typically '00002651_L_500_plain_06.png'.
        We extract the numeric part '00002651'.
        """
        match = re.match(r"^(\d+)", filepath.stem)
        if match:
            return match.group(1)
        return None

    def extract_lfw_id(self, filepath: Path) -> str | None:
        """
        Extracts the person ID from the LFW directory structure.
        The format is '.../lfw/George_W_Bush/George_W_Bush_0001.jpg'.
        The ID is the parent directory name.
        """
        # The ID is the name of the parent directory.
        return filepath.parent.name

    def scan_directory(
        self,
        dir_path: Path,
        id_extractor: callable,
        data_dict: defaultdict,
        dataset_name: str,
    ):
        """Generic function to scan a directory."""
        print(f"Scanning {dataset_name} directory: {dir_path}")
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Warning: {dataset_name} directory not found at {dir_path}")
            return

        file_count = 0
        for filepath in dir_path.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in self.image_extensions:
                file_count += 1
                person_id = id_extractor(filepath)
                if person_id:
                    data_dict[person_id].append(str(filepath))
                else:
                    print(
                        f"Warning: Could not extract ID from {dataset_name} file: {filepath}"
                    )

        print(f"  Processed {file_count} image files.")
        print(f"  Found {len(data_dict)} unique {dataset_name} person IDs.")

    def create_metadata_csv(self):
        """Create the metadata mapping template CSV file."""
        print(f"\nCreating metadata mapping template: {self.output_file}")

        # We will use both NIST and LFW IDs to create the template,
        # giving preference to NIST for creating the initial `humanitas_id`.
        sorted_nist_ids = sorted(self.nist_data.keys())
        sorted_lfw_ids = sorted(self.lfw_data.keys())

        # Combine keys to ensure all individuals are listed
        
        # In this template, we can't auto-match, so we list them separately
        # and let the researcher do the mapping.

        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "humanitas_id",
                "nist_id",
                "lfw_id",
                "nist_file_count",
                "lfw_file_count",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            humanitas_counter = 1

            # Write NIST data
            for nist_id in sorted_nist_ids:
                row = {
                    "humanitas_id": f"person_{humanitas_counter:04d}",
                    "nist_id": nist_id,
                    "lfw_id": "",  # Left blank for manual mapping
                    "nist_file_count": len(self.nist_data[nist_id]),
                    "lfw_file_count": "",
                }
                writer.writerow(row)
                humanitas_counter += 1

            # Add a separator for clarity in the CSV
            writer.writerow({"humanitas_id": "--- LFW DATA BELOW ---"})

            # Write LFW data for reference (to make manual mapping easier)
            for lfw_id in sorted_lfw_ids:
                row = {
                    "humanitas_id": "",
                    "nist_id": "",
                    "lfw_id": lfw_id,
                    "nist_file_count": "",
                    "lfw_file_count": len(self.lfw_data[lfw_id]),
                }
                writer.writerow(row)

        print(f"Template CSV created successfully at: {self.output_file}")
        print(
            "Please open the CSV and manually map 'lfw_id' values to the corresponding 'nist_id' rows."
        )

    def run(self):
        """Execute the complete scanning and mapping process."""
        print("Starting Dataset Metadata and Mapping Script (v1.1)")
        print("-" * 50)

        self.scan_directory(self.nist_dir, self.extract_nist_id, self.nist_data, "NIST")
        self.scan_directory(self.lfw_dir, self.extract_lfw_id, self.lfw_data, "LFW")

        self.create_metadata_csv()

        print("\nScript completed successfully!")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Generate metadata mapping template for HUMANITAS datasets."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory of the project (default: current directory)",
    )
    args = parser.parse_args()

    scanner = DatasetScanner(base_dir=args.base_dir)
    scanner.run()


if __name__ == "__main__":
    main()
