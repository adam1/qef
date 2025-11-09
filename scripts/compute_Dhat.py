#!/usr/bin/env python3
"""
Compute D̂ = Σ_{E,F} B̂_{λ,E,F} for all (E,F) pairs in P_{9,1}.

This script computes the sum of all B̂_{λ,E,F} matrices over the basis P_{n,t}:
    D̂ = Σ_{E,F ∈ P_{n,t}} B̂_{λ,E,F}

where B̂_{λ,E,F} = E†F - λ(E,F)·I.

The result is a symbolic matrix written to disk in sparse format.

Usage:
    python compute_Dhat.py -o output_file [options]

Arguments:
    -o, --output: Output file for D̂ matrix
    -n, --n_qubits: Number of qubits (default: 9)
    -t, --max_weight: Maximum weight t for P_{n,t} basis (default: 1)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import Matrix, simplify
from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.states import create_shor_logical_zero
from qef.error_forms import compute_B_hat_lambda_EF


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}", flush=True)


def run_id():
    """Generate a run ID (timestamp + PID)."""
    return f"{int(datetime.now().timestamp())}_{os.getpid()}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute D̂ = Σ_{E,F} B̂_{λ,E,F} for all pairs in P_{n,t}"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file for D̂ matrix'
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

    args = parser.parse_args()

    # Create output directory if needed
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"Computing D̂ = Σ_{{E,F}} B̂_{{λ,E,F}}")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Output file: {output_file}")
    log("=" * 70)
    log("")

    # Create reference state for λ computation
    log("Creating Shor code logical zero state...")
    ket_0L = create_shor_logical_zero()
    log("  |0_L⟩ created")
    log("")

    # Get the basis P_{n,t}
    log(f"Constructing basis P_{{{args.n_qubits},{args.max_weight}}}...")
    basis = get_basis_P_n_t(args.n_qubits, args.max_weight)
    basis_size = len(basis)
    log(f"Basis size: {basis_size}")
    log("")

    total_pairs = basis_size * basis_size
    pair_count = 0

    # Initialize D̂ as zero matrix
    dim = 2 ** args.n_qubits
    log(f"Initializing D̂ as {dim}×{dim} zero matrix...")
    D_hat = Matrix.zeros(dim, dim)
    log("")

    log("Computing sum over all (E,F) pairs...")
    log("")

    # Loop over all (E, F) pairs and accumulate
    for E_idx in basis:
        for F_idx in basis:
            pair_count += 1

            E_str = index_to_pauli_string(E_idx, args.n_qubits)
            F_str = index_to_pauli_string(F_idx, args.n_qubits)

            if pair_count % 50 == 0:
                log(f"[{pair_count}/{total_pairs}] Processing E={E_idx} ({E_str}), F={F_idx} ({F_str})")

            # Compute B̂_{λ,E,F} and add to sum
            B_hat = compute_B_hat_lambda_EF(E_idx, F_idx, args.n_qubits, ket_0L)
            D_hat += B_hat

    log("")
    log(f"All {total_pairs} pairs processed!")
    log("")

    # Simplify the result
    log("Simplifying matrix entries...")
    log("(This may take some time for symbolic expressions)")
    log("")

    for i in range(dim):
        if i % 50 == 0:
            log(f"Simplifying row {i}/{dim}...")
        for j in range(dim):
            D_hat[i, j] = simplify(D_hat[i, j])

    log("")
    log("Simplification complete!")
    log("")

    # Write to file in sparse format (only non-zero entries)
    log(f"Writing D̂ to {output_file}...")
    log("(Using sparse format: only non-zero entries)")
    log("")

    nonzero_count = 0
    with open(output_file, 'w') as f:
        # Header
        f.write(f"# D̂ = Σ_{{E,F}} B̂_{{λ,E,F}}\n")
        f.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        f.write(f"# Matrix dimension: {dim}×{dim}\n")
        f.write(f"# Total (E,F) pairs summed: {total_pairs}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Format: row col value (0-indexed, sparse format)\n")
        f.write("\n")

        # Write non-zero entries
        for i in range(dim):
            for j in range(dim):
                if D_hat[i, j] != 0:
                    f.write(f"{i} {j} {D_hat[i, j]}\n")
                    nonzero_count += 1

    log(f"Matrix written successfully!")
    log(f"Non-zero entries: {nonzero_count} / {dim * dim}")
    log(f"Sparsity: {100 * (1 - nonzero_count / (dim * dim)):.2f}%")
    log("")
    log("=" * 70)
    log("Done!")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
