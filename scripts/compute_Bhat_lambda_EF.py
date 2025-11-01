#!/usr/bin/env python3
"""
Compute the matrix B̂_{λ,E,F} for a pair of error operators.

This script computes B̂_{λ,E,F} = E†F - λ(E,F)·I for specified error
operators E and F from the Pauli basis.

Usage:
    python compute_Bhat_lambda_EF.py E_index F_index [options]

Arguments:
    E_index: Index for error operator E in the full Pauli basis
    F_index: Index for error operator F in the full Pauli basis

Options:
    -n, --n_qubits: Number of qubits (default: 9)
    -o, --output: Output file for the matrix (optional, prints to stdout if not specified)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import srepr, simplify
from qef.states import create_shor_logical_zero
from qef.operators import index_to_pauli_string
from qef.error_forms import compute_B_hat_lambda_EF
from qef.lambda_hat import compute_lambda


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute B̂_{λ,E,F} matrix for a pair of error operators"
    )
    parser.add_argument(
        'E_index',
        type=int,
        help='Index for error operator E in the full Pauli basis'
    )
    parser.add_argument(
        'F_index',
        type=int,
        help='Index for error operator F in the full Pauli basis'
    )
    parser.add_argument(
        '-n', '--n_qubits',
        type=int,
        default=9,
        help='Number of qubits (default: 9)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for the matrix (optional)'
    )

    args = parser.parse_args()

    log("=" * 70)
    log(f"Computing B̂_{{λ,E,F}} for E={args.E_index}, F={args.F_index}")
    log(f"Number of qubits: {args.n_qubits}")
    log("=" * 70)
    log("")

    # Get Pauli string representations
    E_str = index_to_pauli_string(args.E_index, args.n_qubits)
    F_str = index_to_pauli_string(args.F_index, args.n_qubits)

    log(f"E = {E_str} (index {args.E_index})")
    log(f"F = {F_str} (index {args.F_index})")
    log("")

    # Create the Shor code state
    log("Creating Shor code logical zero state |0_L⟩...")
    ket_0L = create_shor_logical_zero()
    log("")

    # Compute λ(E, F)
    log("Computing λ(E, F)...")
    lambda_val = compute_lambda(args.E_index, args.F_index, args.n_qubits, ket_0L)
    lambda_simplified = simplify(lambda_val)
    log(f"λ(E, F) = {lambda_simplified}")
    log("")

    # Compute B̂_{λ,E,F}
    log("Computing B̂_{λ,E,F}...")
    B_hat = compute_B_hat_lambda_EF(args.E_index, args.F_index, args.n_qubits, ket_0L)
    log(f"Matrix dimension: {B_hat.shape[0]}×{B_hat.shape[1]}")
    log("")

    # Count non-zero entries
    non_zero_count = 0
    for i in range(B_hat.shape[0]):
        for j in range(B_hat.shape[1]):
            if simplify(B_hat[i, j]) != 0:
                non_zero_count += 1

    log(f"Non-zero entries: {non_zero_count} / {B_hat.shape[0] * B_hat.shape[1]}")
    log("")

    # Output the matrix
    if args.output:
        log(f"Writing to {args.output}...")
        with open(args.output, 'w') as f:
            # Write header
            f.write(f"# B̂_{{λ,E,F}} matrix\n")
            f.write(f"# E = {E_str} (index {args.E_index})\n")
            f.write(f"# F = {F_str} (index {args.F_index})\n")
            f.write(f"# λ(E, F) = {lambda_simplified}\n")
            f.write(f"# Matrix dimension: {B_hat.shape[0]}\n")
            f.write(f"# Timestamp: {timestamp()}\n")
            f.write(f"# Format: row col srepr_value (sparse, only non-zero entries)\n")
            f.write(f"# Total non-zero entries: {non_zero_count}\n")
            f.write("\n")

            # Write non-zero entries
            for i in range(B_hat.shape[0]):
                for j in range(B_hat.shape[1]):
                    val = simplify(B_hat[i, j])
                    if val != 0:
                        f.write(f"{i} {j} {srepr(val)}\n")

        log(f"Successfully wrote {non_zero_count} entries to {args.output}")
    else:
        log("Matrix B̂_{λ,E,F} (sparse format, non-zero entries only):")
        log("")
        for i in range(B_hat.shape[0]):
            for j in range(B_hat.shape[1]):
                val = simplify(B_hat[i, j])
                if val != 0:
                    print(f"{i} {j} {srepr(val)}")

    log("")
    log("=" * 70)
    log("Computation complete!")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
