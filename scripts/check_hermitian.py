#!/usr/bin/env python3
"""
Check if a matrix file is Hermitian.

This script reads a matrix from a file (in sparse format) and checks
if it is Hermitian using symbolic computation.

Usage:
    python check_hermitian.py matrix_file.txt

Arguments:
    matrix_file: Path to matrix file in sparse format (row col value)

The script expects files with format:
    # Comments (lines starting with #)
    row col value
    row col value
    ...

where row and col are 0-indexed integers and value is a symbolic expression.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qef.matrix_utils import is_hermitian
from qef.matrix_io import read_sparse_matrix_symbolic, log


def main():
    parser = argparse.ArgumentParser(
        description="Check if a matrix file is Hermitian"
    )
    parser.add_argument(
        'matrix_file',
        help='Path to matrix file in sparse format'
    )

    args = parser.parse_args()

    matrix_file = Path(args.matrix_file)

    if not matrix_file.exists():
        log(f"ERROR: File not found: {matrix_file}")
        return 1

    log("=" * 70)
    log(f"Checking Hermiticity of matrix")
    log(f"File: {matrix_file}")
    log("=" * 70)
    log("")

    # Read matrix
    M = read_sparse_matrix_symbolic(matrix_file)
    log("")

    # Check if Hermitian
    log("Checking if matrix is Hermitian...")
    log("(This may take some time for large symbolic matrices)")
    log("")

    hermitian = is_hermitian(M)

    log("")
    log("=" * 70)
    if hermitian:
        log("RESULT: Matrix is Hermitian")
    else:
        log("RESULT: Matrix is NOT Hermitian")
    log("=" * 70)

    return 0 if hermitian else 1


if __name__ == "__main__":
    sys.exit(main())
