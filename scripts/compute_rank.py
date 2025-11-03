#!/usr/bin/env python3
"""
Compute the rank of a matrix (λ̂ or D̂).

This script reads a matrix from a sparse text file and computes its rank.
The matrix dimension is read from the comment header.

Usage:
    python compute_rank.py input_file.txt

Arguments:
    input_file: Path to matrix file in sparse format (lambdaHat.txt or Dhat.txt)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qef.matrix_io import extract_dimension_from_file, read_sparse_matrix_symbolic, log


def compute_rank(M):
    """
    Compute the rank of a matrix.

    Args:
        M: SymPy Matrix

    Returns:
        Rank (integer)
    """
    log(f"Computing rank of {M.shape[0]}×{M.shape[1]} matrix...")
    log("  (This may take some time for large matrices)")

    rank = M.rank()

    log(f"Rank computed: {rank}")
    return rank


def main():
    parser = argparse.ArgumentParser(
        description="Compute the rank of a matrix (λ̂ or D̂)"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to matrix file in sparse format'
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        log(f"ERROR: File not found: {input_file}")
        return 1

    # Extract dimension from file
    dimension = extract_dimension_from_file(input_file)
    if dimension is None:
        log(f"ERROR: Could not extract matrix dimension from {input_file}")
        log("Expected comment header with 'Dimension: N' or 'Matrix dimension: N'")
        return 1

    log("=" * 70)
    log(f"Computing rank of matrix")
    log(f"Input file: {input_file}")
    log(f"Matrix dimension: {dimension}×{dimension}")
    log("=" * 70)
    log("")

    # Read the matrix
    M = read_sparse_matrix_symbolic(input_file, dimension)
    log("")

    # Compute rank
    rank = compute_rank(M)
    log("")

    log("=" * 70)
    log(f"RESULT: rank = {rank}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
