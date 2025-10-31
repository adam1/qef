#!/usr/bin/env python3
"""
Compute the numerical rank and signature of a matrix (λ̂ or D̂) using eigenvalue decomposition.

This script reads a matrix from a sparse text file, converts it to numerical
form, and computes the signature (p, q, r) using eigenvalue decomposition:
  p = number of positive eigenvalues
  q = number of negative eigenvalues
  r = nullity (number of zero eigenvalues)

Usage:
    python compute_signature_numerical.py input_file.txt [--tolerance 1e-10]

Arguments:
    input_file: Path to matrix file in sparse format (lambdaHat.txt or Dhat.txt)
    --tolerance: Threshold for considering eigenvalues as zero (default: 1e-10)
"""

import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.linalg import eigh
from sympy import sympify


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


def read_sparse_matrix_numerical(filename, dimension):
    """
    Read a sparse matrix file and convert to numerical numpy array.

    Args:
        filename: Path to sparse matrix file
        dimension: Dimension of the square matrix

    Returns:
        NumPy complex array
    """
    log(f"Reading sparse matrix from {filename}...")

    # Initialize zero matrix
    M = np.zeros((dimension, dimension), dtype=complex)

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

            # Convert symbolic expression to complex number
            value_symbolic = sympify(parts[2])
            value_numerical = complex(value_symbolic.evalf())

            M[row, col] = value_numerical
            entry_count += 1

            if entry_count % 10000 == 0:
                log(f"  Read {entry_count} entries...")

    log(f"Read {entry_count} non-zero entries")
    log(f"Converted to {dimension}×{dimension} complex array")

    return M


def compute_signature_eigenvalues(M, tolerance=1e-10):
    """
    Compute signature (p, q, r) using eigenvalue decomposition.

    For a Hermitian matrix M, computes eigenvalues and classifies them
    as positive, negative, or zero.

    Args:
        M: NumPy array (should be Hermitian)
        tolerance: Threshold for considering eigenvalues as zero

    Returns:
        Tuple of (p, q, r, eigenvalues) where:
          p = number of positive eigenvalues
          q = number of negative eigenvalues
          r = nullity (number of zero eigenvalues)
          eigenvalues = array of all eigenvalues (real, sorted descending)
    """
    log(f"Computing eigenvalues for {M.shape[0]}×{M.shape[1]} matrix...")
    log(f"  Using tolerance: {tolerance}")

    # Check if matrix is Hermitian (approximately)
    hermitian_error = np.linalg.norm(M - M.conj().T)
    log(f"  Hermitian error: {hermitian_error:.2e}")

    if hermitian_error > 1e-8:
        log(f"  WARNING: Matrix does not appear to be Hermitian!")

    # Compute eigenvalues and eigenvectors (eigh is optimized for Hermitian matrices)
    # Returns eigenvalues in ascending order, and eigenvectors as columns of V
    eigenvalues, eigenvectors = eigh(M)

    # eigh returns real eigenvalues for Hermitian matrices
    # Sort in descending order for display
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    log(f"  Eigenvalue computation complete")
    log(f"  Eigenvalue range: [{np.min(eigenvalues):.2e}, {np.max(eigenvalues):.2e}]")

    # Verify eigendecomposition: M = V D V*
    log(f"  Verifying eigendecomposition: M = V diag(λ) V*...")
    D = np.diag(eigenvalues)
    V = eigenvectors
    M_reconstructed = V @ D @ V.conj().T

    reconstruction_error = np.linalg.norm(M - M_reconstructed)
    log(f"  Reconstruction error ||M - V D V*||: {reconstruction_error:.2e}")

    if reconstruction_error > 1e-8:
        log(f"  WARNING: Large reconstruction error!")
    else:
        log(f"  Eigendecomposition verified successfully")

    # Classify eigenvalues
    p = np.sum(eigenvalues > tolerance)              # Positive eigenvalues
    q = np.sum(eigenvalues < -tolerance)             # Negative eigenvalues
    r = np.sum(np.abs(eigenvalues) <= tolerance)     # Zero eigenvalues (nullity)

    rank = p + q

    log(f"  Signature: (p={p}, q={q}, r={r})")
    log(f"  Rank = p + q = {rank}")
    log(f"  Nullity = r = {r}")
    log(f"  Dimension = p + q + r = {p + q + r}")

    # Report distribution of eigenvalues
    log(f"  Largest 5 eigenvalues: {eigenvalues_sorted[:5]}")
    if p > 0:
        log(f"  Smallest 5 positive eigenvalues: {eigenvalues_sorted[max(0,p-5):p]}")
    if r > 0 and len(eigenvalues_sorted) > rank:
        zero_eigenvalues = np.abs(eigenvalues[np.abs(eigenvalues) <= tolerance])
        if len(zero_eigenvalues) > 0:
            log(f"  Largest 5 'zero' eigenvalues (by abs value): {np.sort(zero_eigenvalues)[::-1][:5]}")
    if q > 0:
        log(f"  Largest 5 negative eigenvalues (by abs value): {np.sort(eigenvalues[eigenvalues < -tolerance])[:5]}")

    return p, q, r, eigenvalues_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Compute numerical rank and signature using eigenvalue decomposition"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to matrix file in sparse format'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-10,
        help='Threshold for considering eigenvalues as zero (default: 1e-10)'
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
    log(f"Computing signature using eigenvalue decomposition")
    log(f"Input file: {input_file}")
    log(f"Matrix dimension: {dimension}×{dimension}")
    log("=" * 70)
    log("")

    # Read the matrix
    M = read_sparse_matrix_numerical(input_file, dimension)
    log("")

    # Compute signature
    p, q, r, eigenvalues = compute_signature_eigenvalues(M, tolerance=args.tolerance)
    log("")

    log("=" * 70)
    log(f"RESULTS:")
    log(f"  Signature: ({p}, {q}, {r})")
    log(f"  p = {p} (positive eigenvalues)")
    log(f"  q = {q} (negative eigenvalues)")
    log(f"  r = {r} (zero eigenvalues / nullity)")
    log(f"  Rank = p + q = {p + q}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
