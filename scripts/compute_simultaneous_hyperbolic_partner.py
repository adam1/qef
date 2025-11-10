#!/usr/bin/env python3
"""
Compute w solving the simultaneous system B_{λ,E_i,E_j}(u,w) = 1 for all pairs.

This script implements the simultaneous hyperbolic pairing approach:
Instead of solving u†D̂w = 1, we solve all equations simultaneously:
    u†B̂_{λ,E_i,E_j}w = 1 for all (E_i,E_j) in P_{n,t}

This creates a stacked system Q w = [1,1,...,1]^T where:
- Q is (basis_size^2 × 2^n) matrix
- Each row q_{i,j} = u†B̂_{λ,E_i,E_j}

For P_{9,1}: 784 equations, 512 unknowns (overdetermined system)

Usage:
    python compute_simultaneous_hyperbolic_partner.py -o w_vector.txt

Arguments:
    -o, --output: Output file for w vector
    -n, --n_qubits: Number of qubits (default: 9)
    -t, --max_weight: Maximum weight t for P_{n,t} basis (default: 1)
    -u, --use_logical_one: Use |1_L⟩ instead of |0_L⟩ as starting vector
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qef.matrix_io import log
from qef.states import create_shor_logical_zero, create_shor_logical_one
from qef.operators import get_basis_P_n_t
from qef.isotropic_extension import find_simultaneous_hyperbolic_partner


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Compute w solving simultaneous system B_{λ,E,F}(u,w) = 1 for all pairs"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file for w vector'
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
        '-u', '--use_logical_one',
        action='store_true',
        help='Use |1_L⟩ instead of |0_L⟩ as starting vector'
    )

    args = parser.parse_args()

    output_file = Path(args.output)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("Computing Simultaneous Hyperbolic Partner")
    log("Solving: B_{λ,E_i,E_j}(u,w) = 1 for all (E_i,E_j) pairs")
    log("=" * 70)
    log("")
    log(f"Output file: {output_file}")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log("")

    # Create starting isotropic vector
    if args.use_logical_one:
        log("Creating |1_L⟩ as starting vector u...")
        u = create_shor_logical_one()
        u_name = "|1_L⟩"
    else:
        log("Creating |0_L⟩ as starting vector u...")
        u = create_shor_logical_zero()
        u_name = "|0_L⟩"

    log(f"  {u_name} created")
    log("")

    # Get error basis
    log(f"Getting error basis P_{{{args.n_qubits},{args.max_weight}}}...")
    basis = get_basis_P_n_t(args.n_qubits, args.max_weight)
    basis_size = len(basis)
    log(f"  Basis size: {basis_size}")
    log(f"  Number of constraints: {basis_size * basis_size}")
    log("")

    # Solve simultaneous system
    log("Solving simultaneous system...")
    log("")

    try:
        w = find_simultaneous_hyperbolic_partner(
            u, basis, args.n_qubits, u, verbose=True
        )
    except ValueError as e:
        log("")
        log("=" * 70)
        log("ERROR")
        log("=" * 70)
        log(f"  {e}")
        return 1

    log("")
    log("=" * 70)
    log("SUCCESS!")
    log("=" * 70)
    log("")
    log("Solution found for simultaneous system")
    log("")

    # Count nonzero entries in w
    dim = w.shape[0]
    nonzero_count = sum(1 for i in range(dim) if w[i, 0] != 0)
    log(f"Vector w properties:")
    log(f"  Dimension: {dim}")
    log(f"  Non-zero entries: {nonzero_count}")
    log(f"  Sparsity: {100 * (1 - nonzero_count / dim):.2f}%")
    log("")

    # Write w to file
    log(f"Writing w to {output_file}...")

    with open(output_file, 'w') as f:
        # Header
        f.write(f"# Simultaneous hyperbolic partner w\n")
        f.write(f"# Solves: B_{{λ,E_i,E_j}}(u,w) = 1 for all (E_i,E_j) pairs\n")
        f.write(f"# Starting vector: {u_name}\n")
        f.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        f.write(f"# Constraints: {basis_size * basis_size}\n")
        f.write(f"# Dimension: {dim}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Format: index value (0-indexed, sparse format)\n")
        f.write("\n")

        # Write non-zero entries
        for i in range(dim):
            val = w[i, 0]
            if val != 0:
                f.write(f"{i} {val}\n")

    log(f"  Vector written successfully")
    log("")

    log("=" * 70)
    log("Done!")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
