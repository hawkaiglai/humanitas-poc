# HUMANITAS: A Proof-of-Concept for Privacy-Preserving Biometric Identity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-research_prototype-red.svg)](SECURITY.md)

This repository contains the source code for the HUMANITAS Proof-of-Concept (POC), a scientific instrument to validate a novel system for **Sybil-resistant digital identity** using multi-modal biometric fusion and Zero-Knowledge Proofs.

This is the reference implementation for our upcoming academic paper.

## Key Results
- ✅ **False Match Rate (FMR): 0.0%**
- 🔬 **False Non-Match Rate (FNMR):** 83.3% (demonstrating a strict, high-security default threshold)

## Getting Started

1.  **Install:** `poetry install`
2.  **Configure:** `cp .env.example .env` and edit `.env` with your dataset paths.
3.  **Prepare Data:** Run the scripts in the `scripts/` directory.
4.  **Run Tests:** `poetry run python -m humanitas_poc.cli test --max-samples 200`

## How to Cite
Please see the `CITATION.cff` file for citation information.

## License
This project is licensed under the MIT License.
