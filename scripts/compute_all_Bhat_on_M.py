#!/usr/bin/env python3
"""
Compute B̂_{λ,E,F} restricted to code space M for all (E,F) pairs in P_{9,1}.

For each (E,F) pair in P_{9,1}, this script:
1. Computes the full B̂_{λ,E,F} = E†F - λ(E,F)·I matrix
2. Restricts it to the code space M spanned by {|0_L⟩, |1_L⟩}
3. Computes the 2×2 matrix with entries ⟨ψ_i|B̂_{λ,E,F}|ψ_j⟩
4. Verifies that this matrix is zero (expected for quantum error correction)

Usage:
    python compute_all_Bhat_on_M.py -o output_dir [options]

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

from sympy import simplify
from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.states import create_shor_logical_zero, create_shor_logical_one
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
        description="Compute B̂_{λ,E,F} restricted to code space M for all pairs in P_{n,t}"
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
    log(f"Computing B̂_{{λ,E,F}} restricted to code space M")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Output directory: {output_dir}")
    log(f"Run ID: {run_tag}")
    log("=" * 70)
    log("")

    # Create code space basis states
    log("Creating Shor code basis states...")
    ket_0L = create_shor_logical_zero()
    ket_1L = create_shor_logical_one()
    log("  |0_L⟩ created")
    log("  |1_L⟩ created")
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
    summary_file = output_dir / f"Bhat_on_M_summary_{run_tag}.txt"

    log(f"Creating summary file: {summary_file}")
    log("")

    # Track violations
    violations = []

    with open(summary_file, 'w') as summary:
        summary.write(f"# B̂_{{λ,E,F}} restricted to code space M (symbolic computation)\n")
        summary.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        summary.write(f"# Total pairs: {total_pairs}\n")
        summary.write(f"# Run ID: {run_tag}\n")
        summary.write(f"# Timestamp: {timestamp()}\n")
        summary.write(f"# Format: E_index F_index E_str F_str num_nonzero_entries status\n")
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

                # Restrict to code space: compute 2×2 matrix with entries ⟨ψ_i|B̂|ψ_j⟩
                # where ψ_0 = |0_L⟩, ψ_1 = |1_L⟩
                code_basis = [ket_0L, ket_1L]
                B_on_M = [[None, None], [None, None]]

                for i in range(2):
                    for j in range(2):
                        bra_i = code_basis[i].H
                        ket_j = code_basis[j]
                        entry = (bra_i * B_hat_full * ket_j)[0, 0]
                        B_on_M[i][j] = simplify(entry)

                # Check if all entries are exactly zero (symbolically)
                nonzero_count = 0
                nonzero_entries = []
                for i in range(2):
                    for j in range(2):
                        if B_on_M[i][j] != 0:
                            nonzero_count += 1
                            nonzero_entries.append((i, j, B_on_M[i][j]))

                # Determine status
                if nonzero_count == 0:
                    status = "OK"
                else:
                    status = "VIOLATION"
                    violations.append((E_idx, F_idx, E_str, F_str, nonzero_entries))
                    log(f"  WARNING: Non-zero entries found! Count: {nonzero_count}")
                    for i, j, val in nonzero_entries:
                        log(f"    B_M[{i},{j}] = {val}")

                # Write to summary
                summary.write(f"{E_idx} {F_idx} {E_str} {F_str} {nonzero_count} {status}\n")
                summary.flush()

    log("")
    log("=" * 70)
    log("All pairs processed!")
    log("")
    log(f"Total pairs checked: {total_pairs}")
    log(f"Violations found: {len(violations)}")
    log("")

    if violations:
        log("Violations:")
        for E_idx, F_idx, E_str, F_str, nonzero_entries in violations:
            log(f"  E={E_idx} ({E_str}), F={F_idx} ({F_str}):")
            for i, j, val in nonzero_entries:
                log(f"    B_M[{i},{j}] = {val}")
        log("")
    else:
        log("SUCCESS: All B̂_{λ,E,F} matrices are zero on code space M!")
        log("")

    log(f"Summary written to: {summary_file}")
    log("=" * 70)

    return 0 if len(violations) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
