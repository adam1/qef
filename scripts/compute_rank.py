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
import re
from pathlib import Path
from datetime import datetime

from sympy import Matrix, sympify


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}", flush=True)


def extract_dimension_from_file(filename):
    """
    Extract matrix dimension from comment header.

    Looks for patterns like:
    - # Dimension: 28
    - # Matrix dimension: 512

    Args:
        filename: Path to matrix file

    Returns:
        Dimension (integer) or None if not found
    """
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                # Past the header
                break

            # Look for "Dimension: N" or "Matrix dimension: N"
            match = re.search(r'(?:Matrix\s+)?[Dd]imension:\s*(\d+)', line)
            if match:
                return int(match.group(1))

    return None


def read_sparse_matrix(filename, dimension):
    """
    Read a sparse matrix file and reconstruct the full matrix.

    Args:
        filename: Path to sparse matrix file
        dimension: Dimension of the square matrix

    Returns:
        SymPy Matrix
    """
    log(f"Reading sparse matrix from {filename}...")

    # Initialize zero matrix
    M = Matrix.zeros(dimension, dimension)

    entry_count = 0
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            # Parse: row col srepr_value
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue

            row = int(parts[0])
            col = int(parts[1])
            value = sympify(parts[2])

            M[row, col] = value
            entry_count += 1

            if entry_count % 10000 == 0:
                log(f"  Read {entry_count} entries...")

    log(f"Read {entry_count} non-zero entries")
    return M


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
    M = read_sparse_matrix(input_file, dimension)
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
