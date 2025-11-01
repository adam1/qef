#!/usr/bin/env python3
"""
Compute B̂_{λ,E,F} matrices and their signatures for all (E,F) pairs in P_{9,1}.

This script:
1. Loops over all (E,F) pairs in the P_{9,1} basis (28×28 = 784 pairs)
2. For each pair, computes B̂_{λ,E,F} using compute_Bhat_lambda_EF.py
3. For each matrix, computes the signature using compute_signature_numerical.py
4. Captures all logs to timestamped files

Usage:
    python compute_all_Bhat_signatures.py -o output_dir [options]

Arguments:
    -o, --output: Output directory for matrices and logs
    -n, --n_qubits: Number of qubits (default: 9)
    -t, --max_weight: Maximum weight t for P_{n,t} basis (default: 1)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qef.operators import get_basis_P_n_t, index_to_pauli_string


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
        description="Compute B̂_{λ,E,F} signatures for all pairs in P_{n,t}"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for matrices and logs'
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

    # Get scripts directory
    scripts_dir = Path(__file__).parent

    log("=" * 70)
    log(f"Computing B̂_{{λ,E,F}} signatures for all pairs in P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Output directory: {output_dir}")
    log(f"Run ID: {run_tag}")
    log("=" * 70)
    log("")

    # Get the basis P_{n,t}
    log(f"Constructing basis P_{{{args.n_qubits},{args.max_weight}}}...")
    basis = get_basis_P_n_t(args.n_qubits, args.max_weight)
    basis_size = len(basis)
    log(f"Basis size: {basis_size}")
    log("")

    total_pairs = basis_size * basis_size
    pair_count = 0

    # Summary file for all signatures
    summary_file = output_dir / f"Bhat_signatures_summary_{run_tag}.txt"

    log(f"Creating summary file: {summary_file}")
    log("")

    with open(summary_file, 'w') as summary:
        summary.write(f"# B̂_{{λ,E,F}} Signature Summary\n")
        summary.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        summary.write(f"# Total pairs: {total_pairs}\n")
        summary.write(f"# Run ID: {run_tag}\n")
        summary.write(f"# Timestamp: {timestamp()}\n")
        summary.write(f"# Format: E_index F_index E_str F_str p q r\n")
        summary.write("\n")

        # Loop over all (E, F) pairs
        for E_idx in basis:
            for F_idx in basis:
                pair_count += 1

                E_str = index_to_pauli_string(E_idx, args.n_qubits)
                F_str = index_to_pauli_string(F_idx, args.n_qubits)

                log(f"[{pair_count}/{total_pairs}] Processing E={E_idx} ({E_str}), F={F_idx} ({F_str})")

                # Output file for B̂_{λ,E,F} matrix
                matrix_file = output_dir / f"Bhat_{E_idx}_{F_idx}.txt"

                # Log file for B̂ computation
                bhat_log = output_dir / f"Bhat_{E_idx}_{F_idx}_{run_tag}.log"

                # Compute B̂_{λ,E,F}
                log(f"  Computing B̂_{{λ,E,F}}...")
                bhat_cmd = [
                    "python3.11",
                    str(scripts_dir / "compute_Bhat_lambda_EF.py"),
                    str(E_idx),
                    str(F_idx),
                    "-n", str(args.n_qubits),
                    "-o", str(matrix_file)
                ]

                with open(bhat_log, 'w') as log_file:
                    result = subprocess.run(
                        bhat_cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True
                    )

                if result.returncode != 0:
                    log(f"  ERROR: B̂_{{λ,E,F}} computation failed for E={E_idx}, F={F_idx}")
                    log(f"  See log: {bhat_log}")
                    continue

                # Log file for signature computation
                sig_log = output_dir / f"signature_Bhat_{E_idx}_{F_idx}_{run_tag}.log"

                # Compute signature
                log(f"  Computing signature...")
                sig_cmd = [
                    "python3.11",
                    str(scripts_dir / "compute_signature_numerical.py"),
                    str(matrix_file)
                ]

                with open(sig_log, 'w') as log_file:
                    result = subprocess.run(
                        sig_cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        capture_output=False
                    )

                if result.returncode != 0:
                    log(f"  ERROR: Signature computation failed for E={E_idx}, F={F_idx}")
                    log(f"  See log: {sig_log}")
                    continue

                # Parse signature from log file
                p, q, r = None, None, None
                with open(sig_log, 'r') as f:
                    for line in f:
                        if "p = " in line and "positive eigenvalues" in line:
                            p = int(line.split("p = ")[1].split()[0])
                        elif "q = " in line and "negative eigenvalues" in line:
                            q = int(line.split("q = ")[1].split()[0])
                        elif "r = " in line and "zero eigenvalues" in line:
                            r = int(line.split("r = ")[1].split()[0])

                if p is not None and q is not None and r is not None:
                    log(f"  Signature: ({p}, {q}, {r})")
                    summary.write(f"{E_idx} {F_idx} {E_str} {F_str} {p} {q} {r}\n")
                    summary.flush()
                else:
                    log(f"  WARNING: Could not parse signature from log")

                log("")

    log("=" * 70)
    log("All pairs processed successfully!")
    log(f"Summary written to: {summary_file}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
