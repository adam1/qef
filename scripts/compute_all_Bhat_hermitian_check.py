#!/usr/bin/env python3
"""
Check if B̂_{λ,E,F} is Hermitian for all (E,F) pairs in P_{9,1}.

For each (E,F) pair in P_{9,1}, this script:
1. Computes the full B̂_{λ,E,F} = E†F - λ(E,F)·I matrix
2. Checks if the matrix is Hermitian (symbolically)
3. Reports summary counts of Hermitian vs non-Hermitian matrices

Usage:
    python compute_all_Bhat_hermitian_check.py -o output_dir [options]

Arguments:
    -o, --output: Output directory for logs and summary
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

from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.states import create_shor_logical_zero
from qef.error_forms import compute_B_hat_lambda_EF
from qef.matrix_utils import is_hermitian


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
        description="Check if B̂_{λ,E,F} is Hermitian for all pairs in P_{n,t}"
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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID for this session
    run_tag = run_id()

    log("=" * 70)
    log(f"Checking Hermiticity of B̂_{{λ,E,F}} matrices")
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

    total_pairs = basis_size * basis_size
    pair_count = 0

    # Summary file
    summary_file = output_dir / f"Bhat_hermitian_summary_{run_tag}.txt"

    log(f"Creating summary file: {summary_file}")
    log("")

    # Track results
    hermitian_pairs = []
    non_hermitian_pairs = []

    with open(summary_file, 'w') as summary:
        summary.write(f"# B̂_{{λ,E,F}} Hermiticity check (symbolic computation)\n")
        summary.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        summary.write(f"# Total pairs: {total_pairs}\n")
        summary.write(f"# Run ID: {run_tag}\n")
        summary.write(f"# Timestamp: {timestamp()}\n")
        summary.write(f"# Format: E_index F_index E_str F_str is_hermitian\n")
        summary.write("\n")

        # Loop over all (E, F) pairs
        for E_idx in basis:
            for F_idx in basis:
                pair_count += 1

                E_str = index_to_pauli_string(E_idx, args.n_qubits)
                F_str = index_to_pauli_string(F_idx, args.n_qubits)

                if pair_count % 50 == 0:
                    log(f"[{pair_count}/{total_pairs}] Processing E={E_idx} ({E_str}), F={F_idx} ({F_str})")

                # Compute full B̂_{λ,E,F} (512×512 for 9 qubits)
                B_hat_full = compute_B_hat_lambda_EF(E_idx, F_idx, args.n_qubits, ket_0L)

                # Check if Hermitian
                hermitian = is_hermitian(B_hat_full)

                # Track results
                if hermitian:
                    hermitian_pairs.append((E_idx, F_idx, E_str, F_str))
                else:
                    non_hermitian_pairs.append((E_idx, F_idx, E_str, F_str))

                # Write to summary
                summary.write(f"{E_idx} {F_idx} {E_str} {F_str} {hermitian}\n")
                summary.flush()

    log("")
    log("=" * 70)
    log("All pairs processed!")
    log("")
    log(f"Total pairs checked: {total_pairs}")
    log(f"Hermitian matrices: {len(hermitian_pairs)}")
    log(f"Non-Hermitian matrices: {len(non_hermitian_pairs)}")
    log("")

    log(f"Summary written to: {summary_file}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
