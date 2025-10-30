#!/usr/bin/env python3
"""
Merge partial D̂ matrix files into a single matrix file.

This script reads multiple partial matrix files and combines them into a single
output file containing the complete D̂ matrix in sparse format.
"""

import argparse
from datetime import datetime
from sympy import sympify, conjugate, Matrix


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}")


def read_partial_file(filename):
    """
    Read a partial matrix file.

    Args:
        filename: Path to partial matrix file

    Returns:
        Dict mapping (row, col) -> sympy expression
    """
    entries = {}
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue

            # Parse: row col srepr_value
            parts = line.split(None, 2)  # Split on whitespace, max 3 parts
            if len(parts) != 3:
                continue

            row = int(parts[0])
            col = int(parts[1])
            value = sympify(parts[2])

            entries[(row, col)] = value

    return entries


def merge_partial_files(partial_files):
    """
    Merge multiple partial files into a single entries dict.

    Args:
        partial_files: List of filenames to merge

    Returns:
        Dict mapping (row, col) -> sympy expression
    """
    merged = {}

    for filename in partial_files:
        log(f"Reading {filename}...")
        partial = read_partial_file(filename)
        log(f"  Found {len(partial)} entries")

        # Check for duplicates
        for key in partial:
            if key in merged:
                log(f"  WARNING: Duplicate entry at {key}, overwriting")

        merged.update(partial)

    return merged


def fill_hermitian(entries, matrix_dim):
    """
    Fill in the lower triangular part of D̂ using Hermitian property.

    For entries in upper triangle (i, j) where i < j, add (j, i) = conj((i, j)).

    Args:
        entries: Dict mapping (row, col) -> value (upper triangular)
        matrix_dim: Dimension of the matrix

    Returns:
        Dict with both upper and lower triangular entries
    """
    full_entries = entries.copy()

    for (i, j), value in entries.items():
        if i < j:
            # Add lower triangular entry
            full_entries[(j, i)] = conjugate(value)

    return full_entries


def write_merged_file(entries, matrix_dim, output_file, n_qubits, max_weight):
    """
    Write merged entries to output file.

    Args:
        entries: Dict mapping (row, col) -> value
        matrix_dim: Dimension of the matrix
        output_file: Output filename
        n_qubits: Number of qubits
        max_weight: Maximum weight t for P_{n,t} basis
    """
    log(f"Writing merged matrix to {output_file}...")

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# D̂ matrix (total error form) for Shor code\n")
        f.write(f"# Number of qubits: {n_qubits}\n")
        f.write(f"# Basis: P_{{{n_qubits},{max_weight}}}\n")
        f.write(f"# Matrix dimension: {matrix_dim}×{matrix_dim} (computational basis)\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Format: row col srepr_value (sparse, only non-zero entries)\n")
        f.write(f"# Total entries: {len(entries)}\n")

        # Write entries in sorted order (row, col)
        for (i, j) in sorted(entries.keys()):
            value = entries[(i, j)]
            f.write(f"{i} {j} {value.__repr__()}\n")

    log(f"Wrote {len(entries)} entries")


def main():
    parser = argparse.ArgumentParser(
        description='Merge partial D̂ matrix files'
    )
    parser.add_argument(
        'partial_files',
        nargs='+',
        help='Partial matrix files to merge'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='Dhat.txt',
        help='Output file (default: Dhat.txt)'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=512,
        help='Matrix dimension (default: 512 for 9 qubits)'
    )
    parser.add_argument(
        '-n', '--n_qubits',
        type=int,
        default=9,
        help='Number of qubits (default: 9)'
    )
    parser.add_argument(
        '-t', '--max_weight',
        type=int,
        default=1,
        help='Maximum weight t for P_{n,t} basis (default: 1)'
    )
    parser.add_argument(
        '--no-fill-hermitian',
        action='store_true',
        help='Do not fill lower triangular part (keep only upper triangle)'
    )

    args = parser.parse_args()

    log("=" * 70)
    log("Merging partial D̂ matrix files")
    log("=" * 70)
    print()

    # Read and merge all partial files
    entries = merge_partial_files(args.partial_files)
    log(f"Total entries read: {len(entries)}")
    print()

    # Fill in lower triangular part if requested
    if not args.no_fill_hermitian:
        log("Filling lower triangular part using Hermitian property...")
        entries = fill_hermitian(entries, args.dimension)
        log(f"Total entries after filling: {len(entries)}")
        print()

    # Write merged file
    write_merged_file(entries, args.dimension, args.output, args.n_qubits, args.max_weight)
    print()

    log("=" * 70)
    log("Merge complete!")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
