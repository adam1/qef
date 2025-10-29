#!/usr/bin/env python3
"""
Compute the lambda-hat matrix λ̂ in parallel using multiprocessing.

This script computes the λ̂ matrix for the Shor code, where λ̂[i,j] = λ(E_i, E_j)
for basis elements in P_{9,1}. The matrix is 28×28 and Hermitian.

Workers compute different column ranges in parallel. Since the matrix is Hermitian,
we only compute the upper triangular part (i ≤ j).

Usage:
    python compute_lambdaHat_parallel.py -o output_dir -w 4 -n 9 -t 1

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

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import srepr
from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.lambda_hat import compute_lambda


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    """Print a message with a timestamp."""
    print(f"[{timestamp()}] {message}", flush=True)


def compute_column_range(args):
    """
    Worker function to compute a range of columns of λ̂.

    Since λ̂ is Hermitian, we only compute the upper triangular part.
    For columns [col_start, col_end], we compute entries (i, j) where i ≤ j.

    Args:
        args: Tuple of (worker_id, col_start, col_end, n_qubits, max_weight, output_dir)

    Returns:
        Tuple of (worker_id, output_file_path)
    """
    worker_id, col_start, col_end, n_qubits, max_weight, output_dir = args

    log(f"Worker {worker_id}: Computing columns {col_start}-{col_end}")

    # Get the basis P_{n,t}
    basis = get_basis_P_n_t(n_qubits, max_weight)
    basis_size = len(basis)

    # Create the quantum state |v⟩
    ket_v = create_shor_logical_zero()

    # Store results as list of (row, col, value) tuples
    entries = []

    total_computed = 0

    # For each column in our range
    for j in range(col_start, col_end + 1):
        # Compute entries (i, j) where i <= j (upper triangular)
        for i in range(j + 1):
            s_index = basis[i]
            t_index = basis[j]

            s_str = index_to_pauli_string(s_index, n_qubits)
            t_str = index_to_pauli_string(t_index, n_qubits)

            log(f"Worker {worker_id}: λ̂[{i},{j}]: ({s_str}) × ({t_str})")

            # Compute λ(s, t)
            entry = compute_lambda(s_index, t_index, n_qubits, ket_v)
            entry_simplified = entry.simplify()

            # Only store non-zero entries (sparse format)
            if entry_simplified != 0:
                entries.append((i, j, entry_simplified))

            total_computed += 1

    # Write results to file
    output_file = Path(output_dir) / f"lambdaHat_partial_{worker_id}.txt"

    log(f"Worker {worker_id}: Writing {len(entries)} non-zero entries to {output_file}")

    with open(output_file, "w") as f:
        # Write header
        f.write(f"# Partial lambda-hat matrix λ̂ for Shor code P_{{{n_qubits},{max_weight}}} basis\n")
        f.write(f"# Columns: {col_start} to {col_end}\n")
        f.write(f"# Worker: {worker_id}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Format: row col srepr_value\n")

        for row, col, val in entries:
            f.write(f"{row} {col} {srepr(val)}\n")

    log(f"Worker {worker_id}: Completed columns {col_start}-{col_end} ({total_computed} entries computed)")

    return worker_id, str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute λ̂ matrix in parallel using multiprocessing"
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

    # Get basis to determine matrix dimension
    basis = get_basis_P_n_t(args.n_qubits, args.max_weight)
    dim = len(basis)

    log("=" * 60)
    log(f"Computing λ̂ matrix using {args.workers} workers")
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

        log(f"Worker {worker_id}: columns {col_start}-{col_end}")

    # Create process pool and run workers
    log("")
    log(f"Spawning {args.workers} worker processes...")
    log("")

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
    log("Next step: merge partial files using merge_lambdaHat.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
