#!/usr/bin/env python3.11
"""
Compute the total error form matrix D̂ in parallel using multiprocessing.

This script computes D̂ = ∑_{E,F ∈ basis} D̂_{λ,E,F} where each D̂_{λ,E,F}
is a 512×512 matrix for the 9-qubit Shor code.

Workers compute different column ranges in parallel, accumulating contributions
from all (E,F) pairs. Results are saved in sparse text format.

Usage:
    python compute_Dhat_parallel.py -o output_dir -w 4 -n 9 -t 1

Arguments:
    -o, --output: Output directory for partial files
    -w, --workers: Number of parallel workers (default: 4)
    -n, --n_qubits: Number of qubits (default: 9)
    -t, --max_weight: Maximum weight t for P_{n,t} basis (default: 1)
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import Matrix, simplify, srepr
from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t
from qef.error_forms import compute_D_lambda_EF


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}", flush=True)


def compute_column_range(args):
    """
    Worker function to compute a range of columns of D̂.

    This function computes columns [col_start, col_end] by summing
    D̂_{λ,E,F} over all (E,F) pairs in the basis.

    Args:
        args: Tuple of (worker_id, col_start, col_end, n_qubits, max_weight)

    Returns:
        Tuple of (worker_id, output_file_path)
    """
    worker_id, col_start, col_end, n_qubits, max_weight, output_dir = args

    # Log start
    log(f"Worker {worker_id}: Computing columns {col_start}-{col_end}")

    # Get the basis P_{n,t}
    basis = get_basis_P_n_t(n_qubits, max_weight)
    basis_size = len(basis)

    # Create the quantum state |v⟩
    ket_v = create_shor_logical_zero()

    # Dimension of the Hilbert space
    dim = 2 ** n_qubits

    # Initialize accumulator for this column range
    # We'll accumulate as a dictionary: (row, col) -> value
    # Only store non-zero entries
    accumulator = {}

    # Double loop over all (E, F) pairs
    total_pairs = basis_size * basis_size
    pair_count = 0

    for E_idx in basis:
        for F_idx in basis:
            pair_count += 1

            #if pair_count % 10 == 0:
            log(f"Worker {worker_id}: Processing pair {pair_count}/{total_pairs} (E={E_idx}, F={F_idx})")

            # Compute D̂_{λ,E,F}
            D_lambda_EF = compute_D_lambda_EF(E_idx, F_idx, n_qubits, ket_v)

            # Extract columns [col_start, col_end] and accumulate
            for col in range(col_start, col_end + 1):
                for row in range(dim):
                    val = D_lambda_EF[row, col]

                    # Simplify and check if non-zero
                    val_simplified = simplify(val)
                    if val_simplified != 0:
                        key = (row, col)
                        if key in accumulator:
                            accumulator[key] = simplify(accumulator[key] + val_simplified)
                        else:
                            accumulator[key] = val_simplified

    # Write results to file
    output_file = Path(output_dir) / f"dhat_partial_{worker_id}.txt"

    log(f"Worker {worker_id}: Writing {len(accumulator)} non-zero entries to {output_file}")

    with open(output_file, "w") as f:
        for (row, col), val in sorted(accumulator.items()):
            # Write in sparse format: row col srepr(value)
            f.write(f"{row} {col} {srepr(val)}\n")

    log(f"Worker {worker_id}: Completed columns {col_start}-{col_end}")

    return worker_id, str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute D̂ matrix in parallel using multiprocessing"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for partial files"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "-n", "--n_qubits",
        type=int,
        default=9,
        help="Number of qubits (default: 9)"
    )
    parser.add_argument(
        "-t", "--max_weight",
        type=int,
        default=1,
        help="Maximum weight t for P_{n,t} basis (default: 1)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute matrix dimension
    dim = 2 ** args.n_qubits

    log("=" * 60)
    log(f"Computing D̂ matrix using {args.workers} workers")
    log(f"Matrix dimension: {dim}×{dim}")
    log(f"Number of qubits: {args.n_qubits}")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Output directory: {output_dir}")
    log("=" * 60)

    # Divide columns among workers
    cols_per_worker = dim // args.workers

    worker_args = []
    for worker_id in range(args.workers):
        col_start = worker_id * cols_per_worker

        # Last worker gets remaining columns
        if worker_id == args.workers - 1:
            col_end = dim - 1
        else:
            col_end = col_start + cols_per_worker - 1

        worker_args.append((
            worker_id,
            col_start,
            col_end,
            args.n_qubits,
            args.max_weight,
            str(output_dir)
        ))

    # Create process pool and run workers
    log("")
    log(f"Spawning {args.workers} worker processes...")

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(compute_column_range, worker_args)

    log("")
    log("=" * 60)
    log("All workers completed successfully!")
    log("")
    log("Partial files created:")
    for worker_id, output_file in sorted(results):
        log(f"  Worker {worker_id}: {output_file}")
    log("=" * 60)

    log("")
    log("Next step: merge partial files using merge_Dhat.py")


if __name__ == "__main__":
    main()
