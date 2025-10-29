#!/usr/bin/env python3
"""
Compute a subset of columns of the lambda-hat matrix λ̂ for parallel computation.

This script computes columns col_start through col_end (inclusive) of the
matrix λ̂ and saves the partial result to a text file.

Output format: one line per entry
    row col srepr_value
"""

import argparse
from datetime import datetime
from sympy import srepr

from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.lambda_hat import compute_lambda


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}")


def compute_partial_lambda_hat(basis_indices, n_qubits, ket_v, col_start, col_end, output_file):
    """
    Compute a subset of columns of the lambda-hat matrix λ̂.

    Since λ̂ is Hermitian, we compute the upper triangular part.
    For columns col_start through col_end, we compute entries (i, j) where i <= j.

    Args:
        basis_indices: List of indices in the full Pauli basis
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector
        col_start: First column to compute (inclusive, 0-indexed)
        col_end: Last column to compute (inclusive, 0-indexed)
        output_file: File object to write results to
    """
    matrix_dim = len(basis_indices)

    if col_start < 0 or col_end >= matrix_dim or col_start > col_end:
        raise ValueError(f"Invalid column range [{col_start}, {col_end}] for dimension {matrix_dim}")

    log(f"Computing columns {col_start} through {col_end} (dimension {matrix_dim})")

    total_computed = 0

    # For each column in our range
    for j in range(col_start, col_end + 1):
        # Compute entries (i, j) where i <= j (upper triangular)
        for i in range(j + 1):
            s_index = basis_indices[i]
            t_index = basis_indices[j]

            s_str = index_to_pauli_string(s_index, n_qubits)
            t_str = index_to_pauli_string(t_index, n_qubits)

            log(f"  λ̂[{i},{j}]: ({s_str}) × ({t_str})")

            entry = compute_lambda(s_index, t_index, n_qubits, ket_v)
            entry_simplified = entry.simplify()

            # Only write non-zero entries (sparse format)
            if entry_simplified != 0:
                output_file.write(f"{i} {j} {srepr(entry_simplified)}\n")

            total_computed += 1

    log(f"Computed {total_computed} entries for columns {col_start}-{col_end}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute partial lambda-hat matrix λ̂ for Shor code'
    )
    parser.add_argument(
        'col_start',
        type=int,
        help='First column to compute (0-indexed, inclusive)'
    )
    parser.add_argument(
        'col_end',
        type=int,
        help='Last column to compute (0-indexed, inclusive)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output text file for partial results'
    )

    args = parser.parse_args()

    # Configuration
    n_qubits = 9
    t = 1  # At most 1 qubit affected

    log("=" * 70)
    log(f"Computing partial λ̂ matrix: columns {args.col_start}-{args.col_end}")
    log("=" * 70)
    print()

    # Create the Shor code state
    log("Creating Shor code logical zero state |0_L⟩...")
    ket_0L = create_shor_logical_zero()
    print()

    # Get the basis P_{9,1}
    log(f"Constructing basis P_{{{n_qubits},{t}}}...")
    basis = get_basis_P_n_t(n_qubits, t)
    log(f"  Basis dimension: {len(basis)}")
    print()

    # Compute and write the partial matrix
    with open(args.output, 'w') as f:
        # Write header with metadata
        f.write(f"# Partial lambda-hat matrix λ̂ for Shor code P_{{9,1}} basis\n")
        f.write(f"# Columns: {args.col_start} to {args.col_end}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Format: row col srepr_value\n")

        compute_partial_lambda_hat(basis, n_qubits, ket_0L, args.col_start, args.col_end, f)

    print()
    log(f"Successfully saved to {args.output}")
    print()

    log("=" * 70)
    log("Partial computation complete!")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
