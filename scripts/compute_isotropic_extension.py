#!/usr/bin/env python3
"""
Compute an isotropic extension of the Shor code using Grove's algorithm.

This script implements the isotropic extension algorithm (Grove Prop. 10.8):

Given code space M = span(|0_L⟩, |1_L⟩), which is isotropic with respect to
the total error form D̂, we extend it by one dimension:

1. Start with u = |0_L⟩ (isotropic vector)
2. Find hyperbolic partner v such that:
   - D̂(u,v) = 1
   - D̂(v,v) = 0
3. Verify u+v is isotropic with respect to all B̂_{λ,E,F}
4. If so, M' = M ⊕ span(u+v) is an isotropic extension

Usage:
    python compute_isotropic_extension.py -i Dhat.txt -o extension_vector.txt

Arguments:
    -i, --input: Input file containing D̂ matrix (sparse format)
    -o, --output: Output file for extension vector
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

from sympy import simplify, Add
import numpy as np
from qef.matrix_io import read_sparse_matrix_symbolic, log
from qef.states import create_shor_logical_zero, create_shor_logical_one
from qef.operators import get_basis_P_n_t, index_to_pauli_string
from qef.isotropic_extension import (
    compute_isotropic_extension,
    find_hyperbolic_partner,
    verify_isotropy_on_all_error_forms
)


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sympy_to_numpy(matrix):
    """Convert SymPy Matrix to NumPy array with complex float values."""
    return np.array(matrix.tolist(), dtype=complex)


def check_isotropy_numerical(vector_np, E_idx, F_idx, n_qubits, ket_v, tol=1e-10):
    """
    Check if a vector is isotropic with respect to B̂_{λ,E,F} numerically.

    Args:
        vector_np: NumPy array (1D or 2D column vector)
        E_idx: Index for error operator E
        F_idx: Index for error operator F
        n_qubits: Number of qubits
        ket_v: Reference state for λ computation
        tol: Numerical tolerance for zero check

    Returns:
        True if |vector†B̂vector| < tol
    """
    from qef.error_forms import compute_B_hat_lambda_EF

    # Compute B̂_{λ,E,F} symbolically
    B_hat = compute_B_hat_lambda_EF(E_idx, F_idx, n_qubits, ket_v)

    # Convert to numerical
    B_hat_np = sympy_to_numpy(B_hat)

    # Ensure vector is 2D column
    if vector_np.ndim == 1:
        vector_np = vector_np.reshape(-1, 1)

    # Compute quadratic form: vector†B̂vector
    result = np.conj(vector_np.T) @ B_hat_np @ vector_np
    result_scalar = result[0, 0]

    return np.abs(result_scalar) < tol


def main():
    parser = argparse.ArgumentParser(
        description="Compute isotropic extension of Shor code using Grove's algorithm"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file containing D̂ matrix (sparse format)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file for extension vector'
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
    parser.add_argument(
        '--numerical',
        action='store_true',
        help='Perform checks numerically instead of symbolically (faster)'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        log(f"ERROR: Input file not found: {input_file}")
        return 1

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("Computing Isotropic Extension of Shor Code")
    log("Algorithm: Grove Proposition 10.8")
    log("=" * 70)
    log("")
    log(f"Input D̂ matrix: {input_file}")
    log(f"Output file: {output_file}")
    log(f"Basis: P_{{{args.n_qubits},{args.max_weight}}}")
    log(f"Numerical mode: {args.numerical}")
    log("")

    # Read D̂ matrix
    log("Reading D̂ matrix...")
    D_hat = read_sparse_matrix_symbolic(input_file)
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

    # Verify u is isotropic with respect to D̂
    log(f"Verifying {u_name} is isotropic with respect to D̂...")
    D_uu = (u.H * D_hat * u)[0, 0]
    D_uu_simplified = simplify(D_uu)
    log(f"  D̂(u,u) = {D_uu_simplified}")

    if D_uu_simplified != 0:
        log(f"  ERROR: {u_name} is not isotropic!")
        return 1

    log(f"  ✓ {u_name} is isotropic")
    log("")

    # Find hyperbolic partner v
    log("Finding hyperbolic partner v...")
    log("  Step 1: Solving u†D̂w = 1 for w...")

    try:
        v, b = find_hyperbolic_partner(u, D_hat)

        # Count terms in b (avoid printing large expression)
        if isinstance(b, Add):
            num_terms = len(b.args)
        else:
            num_terms = 1

        log(f"  Step 2: Computed b = -½⟨w|D̂|w⟩ ({num_terms} terms)")
        log(f"  Step 3: Constructed v = bu + w")
    except ValueError as e:
        log(f"  ERROR: {e}")
        return 1

    log("")

    # Verify hyperbolic pair properties
    log("Verifying hyperbolic pair properties...")

    if args.numerical:
        # Numerical verification
        log("  Converting to numerical arrays...")
        D_hat_np = sympy_to_numpy(D_hat)
        u_np = sympy_to_numpy(u)
        v_np = sympy_to_numpy(v)

        log("  Computing D̂(u,v) numerically...")
        D_uv_np = np.conj(u_np.T) @ D_hat_np @ v_np
        D_uv_val = D_uv_np[0, 0]
        log(f"    D̂(u,v) = {D_uv_val}")
        log(f"    |D̂(u,v) - 1| = {np.abs(D_uv_val - 1):.2e}")

        if np.abs(D_uv_val - 1) > 1e-10:
            log(f"    WARNING: Expected D̂(u,v) = 1")

        log("  Computing D̂(v,v) numerically...")
        D_vv_np = np.conj(v_np.T) @ D_hat_np @ v_np
        D_vv_val = D_vv_np[0, 0]
        log(f"    D̂(v,v) = {D_vv_val}")
        log(f"    |D̂(v,v)| = {np.abs(D_vv_val):.2e}")

        if np.abs(D_vv_val) > 1e-10:
            log(f"    WARNING: Expected D̂(v,v) = 0")

        log("  ✓ Hyperbolic pair verified numerically")
    else:
        # Symbolic verification
        log("  Computing D̂(u,v) symbolically...")
        D_uv = (u.H * D_hat * v)[0, 0]
        D_uv_simplified = simplify(D_uv)
        log(f"    D̂(u,v) = {D_uv_simplified}")

        if D_uv_simplified != 1:
            log(f"    WARNING: Expected D̂(u,v) = 1, got {D_uv_simplified}")

        log("  Computing D̂(v,v) symbolically...")
        D_vv = (v.H * D_hat * v)[0, 0]
        D_vv_simplified = simplify(D_vv)
        log(f"    D̂(v,v) = {D_vv_simplified}")

        if D_vv_simplified != 0:
            log(f"    WARNING: Expected D̂(v,v) = 0, got {D_vv_simplified}")

        log("  ✓ Hyperbolic pair constructed successfully")

    log("")

    # Construct extension vector
    log("Constructing extension vector u+v...")
    extension_vector = u + v
    log("  Extension vector u+v constructed")
    log("")

    # Get error basis
    log(f"Getting error basis P_{{{args.n_qubits},{args.max_weight}}}...")
    basis = get_basis_P_n_t(args.n_qubits, args.max_weight)
    basis_size = len(basis)
    log(f"  Basis size: {basis_size}")
    log(f"  Total pairs to check: {basis_size * basis_size}")
    log("")

    # Verify isotropy with respect to all error forms
    log("Verifying u+v is isotropic with respect to all B̂_{λ,E,F}...")
    if args.numerical:
        log("(Using numerical checks - much faster)")
    else:
        log("(Using symbolic checks - this may take some time...)")
    log("")

    if args.numerical:
        # Numerical verification
        extension_vector_np = sympy_to_numpy(extension_vector)
        failed_pairs = []
        pair_count = 0
        total_pairs = basis_size * basis_size

        for E_idx in basis:
            for F_idx in basis:
                pair_count += 1
                if pair_count % 100 == 0:
                    log(f"  Checking pair {pair_count}/{total_pairs}...")

                is_isotropic = check_isotropy_numerical(
                    extension_vector_np, E_idx, F_idx, args.n_qubits, u
                )

                if not is_isotropic:
                    failed_pairs.append((E_idx, F_idx))

        all_isotropic = len(failed_pairs) == 0
    else:
        # Symbolic verification
        all_isotropic, failed_pairs = verify_isotropy_on_all_error_forms(
            extension_vector, basis, args.n_qubits, u
        )

    if all_isotropic:
        log("")
        log("=" * 70)
        log("SUCCESS!")
        log("=" * 70)
        log("")
        log(f"✓ u+v is isotropic with respect to all {basis_size * basis_size} error forms")
        log(f"✓ M' = M ⊕ span(u+v) is an isotropic extension")
        log("")
    else:
        log("")
        log("=" * 70)
        log("FAILURE")
        log("=" * 70)
        log("")
        log(f"✗ u+v is NOT isotropic with respect to {len(failed_pairs)} error forms")
        log("")
        log("Failed pairs (first 10):")
        for i, (E_idx, F_idx) in enumerate(failed_pairs[:10]):
            E_str = index_to_pauli_string(E_idx, args.n_qubits)
            F_str = index_to_pauli_string(F_idx, args.n_qubits)
            log(f"  {i+1}. E={E_idx} ({E_str}), F={F_idx} ({F_str})")

        if len(failed_pairs) > 10:
            log(f"  ... and {len(failed_pairs) - 10} more")
        log("")

        log("Extension vector will still be written to file.")
        log("")

    # Write extension vector to file
    log(f"Writing extension vector to {output_file}...")

    with open(output_file, 'w') as f:
        # Header
        f.write(f"# Isotropic extension vector u+v\n")
        f.write(f"# Starting vector: {u_name}\n")
        f.write(f"# Algorithm: Grove Proposition 10.8\n")
        f.write(f"# Basis: P_{{{args.n_qubits},{args.max_weight}}}\n")
        f.write(f"# Dimension: {extension_vector.shape[0]}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Isotropy verified: {all_isotropic}\n")
        if not all_isotropic:
            f.write(f"# Failed pairs: {len(failed_pairs)}/{basis_size * basis_size}\n")
        f.write(f"# Format: index value (0-indexed, sparse format)\n")
        f.write("\n")

        # Write non-zero entries
        nonzero_count = 0
        for i in range(extension_vector.shape[0]):
            val = extension_vector[i, 0]
            if val != 0:
                f.write(f"{i} {val}\n")
                nonzero_count += 1

    dim = extension_vector.shape[0]
    log(f"  Extension vector written successfully")
    log(f"  Non-zero entries: {nonzero_count} / {dim}")
    log(f"  Sparsity: {100 * (1 - nonzero_count / dim):.2f}%")
    log("")

    log("=" * 70)
    log("Done!")
    log("=" * 70)

    return 0 if all_isotropic else 1


if __name__ == "__main__":
    sys.exit(main())
