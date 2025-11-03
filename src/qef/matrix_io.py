"""
Matrix I/O utilities for reading and writing matrices in sparse format.

This module provides functions for reading and writing matrices from/to files
in sparse format, with both symbolic (SymPy) and numerical (NumPy) support.
"""

import re
from pathlib import Path
from datetime import datetime

import numpy as np
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
    - # Matrix dimension: 512×512

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

            # Look for "Dimension: N" or "Matrix dimension: N" or "Matrix dimension: N×N"
            match = re.search(r'(?:Matrix\s+)?[Dd]imension:\s*(\d+)', line)
            if match:
                return int(match.group(1))

    return None


def read_sparse_matrix_symbolic(filename, dimension=None, verbose=True):
    """
    Read a sparse matrix file and reconstruct the full symbolic matrix.

    Args:
        filename: Path to sparse matrix file
        dimension: Dimension of the square matrix (if None, auto-detect from entries)
        verbose: Whether to print progress messages

    Returns:
        SymPy Matrix
    """
    if verbose:
        log(f"Reading sparse matrix from {filename}...")

    # First pass: determine dimension if not provided
    if dimension is None:
        # Try to extract from header
        dimension = extract_dimension_from_file(filename)

        # If still None, scan entries to find max row/col
        if dimension is None:
            max_row = 0
            max_col = 0
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split(None, 2)
                    if len(parts) == 3:
                        max_row = max(max_row, int(parts[0]))
                        max_col = max(max_col, int(parts[1]))
            dimension = max(max_row, max_col) + 1

        if verbose:
            log(f"  Matrix dimension: {dimension}×{dimension}")

    # Initialize zero matrix
    M = Matrix.zeros(dimension, dimension)

    entry_count = 0
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            # Parse: row col value
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue

            row = int(parts[0])
            col = int(parts[1])
            value = sympify(parts[2])

            M[row, col] = value
            entry_count += 1

            if verbose and entry_count % 10000 == 0:
                log(f"  Read {entry_count} entries...")

    if verbose:
        log(f"Read {entry_count} non-zero entries")

    return M


def read_sparse_matrix_numerical(filename, dimension=None, verbose=True):
    """
    Read a sparse matrix file and convert to numerical numpy array.

    Args:
        filename: Path to sparse matrix file
        dimension: Dimension of the square matrix (if None, auto-detect)
        verbose: Whether to print progress messages

    Returns:
        NumPy complex array
    """
    if verbose:
        log(f"Reading sparse matrix from {filename}...")

    # Determine dimension
    if dimension is None:
        dimension = extract_dimension_from_file(filename)

        # If still None, scan entries to find max row/col
        if dimension is None:
            max_row = 0
            max_col = 0
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split(None, 2)
                    if len(parts) == 3:
                        max_row = max(max_row, int(parts[0]))
                        max_col = max(max_col, int(parts[1]))
            dimension = max(max_row, max_col) + 1

        if verbose:
            log(f"  Matrix dimension: {dimension}×{dimension}")

    # Initialize zero matrix
    M = np.zeros((dimension, dimension), dtype=complex)

    entry_count = 0
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            # Parse: row col value
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

            if verbose and entry_count % 10000 == 0:
                log(f"  Read {entry_count} entries...")

    if verbose:
        log(f"Read {entry_count} non-zero entries")
        log(f"Converted to {dimension}×{dimension} complex array")

    return M
