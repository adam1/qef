#!/usr/bin/env python3
"""
Analyze properties of B̂_{λ,E,F} matrices for all (E,F) pairs in P_{9,1}.

For each (E,F) pair in P_{9,1}, this script:
1. Computes the full B̂_{λ,E,F} = E†F - λ(E,F)·I matrix
2. Checks if the matrix is Hermitian (M† = M)
3. Checks if the matrix is skew-Hermitian (M† = -M)
4. Computes the signature (p, q, r) numerically via eigenvalues
5. Computes the Witt index = min(p, q)
6. Checks if the matrix is singular (rank < dimension)
7. Reports summary statistics

Usage:
    python analyze_all_Bhat_properties.py -o output_dir [options]

Arguments:
    -o, --output: Output directory for logs and summary
    -n, --n_qubits: Number of qubits (default: 9)
    -t, --max_weight: Maximum weight t for P_{n,t} basis (default: 1)
    --tolerance: Tolerance for eigenvalue classification (default: 1e-10)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.linalg import eigh

from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.states import create_shor_logical_zero
from qef.error_forms import compute_B_hat_lambda_EF
from qef.matrix_utils import is_hermitian, is_skew_hermitian, sympy_to_numpy, compute_signature


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
        description="Analyze properties of B̂_{λ,E,F} matrices for all pairs in P_{n,t}"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for logs and summary'
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
        '--tolerance',
        type=float,
        default=1e-10,
        help='Tolerance for eigenvalue classification (default: 1e-10)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID for this session
    run_tag = run_id()

    log("=" * 70)
    log(f"Analyzing properties of B̂_{{λ,E,F}} matrices")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Output directory: {output_dir}")
    log(f"Run ID: {run_tag}")
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

    dim = 2 ** args.n_qubits
    total_pairs = basis_size * basis_size
    pair_count = 0

    # Summary file
    summary_file = output_dir / f"Bhat_properties_summary_{run_tag}.txt"

    log(f"Creating summary file: {summary_file}")
    log("")

    # Track results
    hermitian_count = 0
    skew_hermitian_count = 0
    singular_count = 0
    rank_distribution = {}
    signature_distribution = {}
    witt_index_distribution = {}

    with open(summary_file, 'w') as summary:
        summary.write(f"# B̂_{{λ,E,F}} matrix property analysis\n")
        summary.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        summary.write(f"# Matrix dimension: {dim}×{dim}\n")
        summary.write(f"# Total pairs: {total_pairs}\n")
        summary.write(f"# Run ID: {run_tag}\n")
        summary.write(f"# Timestamp: {timestamp()}\n")
        summary.write(f"# Eigenvalue tolerance: {args.tolerance}\n")
        summary.write(f"# Format: E_idx F_idx E_str F_str hermitian skew_hermitian p q r rank witt_index singular\n")
        summary.write("\n")

        # Loop over all (E, F) pairs
        for E_idx in basis:
            for F_idx in basis:
                pair_count += 1

                E_str = index_to_pauli_string(E_idx, args.n_qubits)
                F_str = index_to_pauli_string(F_idx, args.n_qubits)

                if pair_count % 50 == 0:
                    log(f"[{pair_count}/{total_pairs}] Processing E={E_idx} ({E_str}), F={F_idx} ({F_str})")

                # Compute full B̂_{λ,E,F}
                B_hat_full = compute_B_hat_lambda_EF(E_idx, F_idx, args.n_qubits, ket_0L)

                # Check if Hermitian
                hermitian = is_hermitian(B_hat_full)
                if hermitian:
                    hermitian_count += 1

                # Check if skew-Hermitian
                skew_hermitian = is_skew_hermitian(B_hat_full)
                if skew_hermitian:
                    skew_hermitian_count += 1

                # Convert to numerical for signature computation
                B_hat_np = sympy_to_numpy(B_hat_full)

                # Compute eigenvalues and signature
                eigenvalues = eigh(B_hat_np, eigvals_only=True)
                p, q, r, witt_index = compute_signature(eigenvalues, tolerance=args.tolerance)
                rank = p + q

                # Check if singular
                singular = (rank < dim)
                if singular:
                    singular_count += 1

                # Track distributions
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
                sig_key = (p, q, r)
                signature_distribution[sig_key] = signature_distribution.get(sig_key, 0) + 1
                witt_index_distribution[witt_index] = witt_index_distribution.get(witt_index, 0) + 1

                # Write to summary
                summary.write(f"{E_idx} {F_idx} {E_str} {F_str} {hermitian} {skew_hermitian} {p} {q} {r} {rank} {witt_index} {singular}\n")
                summary.flush()

    log("")
    log("=" * 70)
    log("All pairs processed!")
    log("")
    log(f"Total pairs checked: {total_pairs}")
    log("")
    log("Property counts:")
    log(f"  Hermitian matrices:       {hermitian_count} ({100*hermitian_count/total_pairs:.1f}%)")
    log(f"  Skew-Hermitian matrices:  {skew_hermitian_count} ({100*skew_hermitian_count/total_pairs:.1f}%)")
    log(f"  Singular matrices:        {singular_count} ({100*singular_count/total_pairs:.1f}%)")
    log(f"  Non-singular matrices:    {total_pairs - singular_count} ({100*(total_pairs-singular_count)/total_pairs:.1f}%)")
    log("")
    log("Rank distribution:")
    for rank in sorted(rank_distribution.keys()):
        count = rank_distribution[rank]
        log(f"  Rank {rank}: {count} matrices ({100*count/total_pairs:.1f}%)")
    log("")
    log("Signature distribution (p, q, r):")
    for sig in sorted(signature_distribution.keys()):
        count = signature_distribution[sig]
        log(f"  {sig}: {count} matrices ({100*count/total_pairs:.1f}%)")
    log("")
    log("Witt index distribution:")
    for witt in sorted(witt_index_distribution.keys()):
        count = witt_index_distribution[witt]
        log(f"  Witt index {witt}: {count} matrices ({100*count/total_pairs:.1f}%)")
    log("")

    log(f"Summary written to: {summary_file}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
