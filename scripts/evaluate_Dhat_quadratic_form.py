#!/usr/bin/env python3
"""
Evaluate the quadratic form Q associated with D̂ on code space basis vectors.

For the Hermitian form D with matrix D̂, the quadratic form is:
    Q(v) = ⟨v|D̂|v⟩

This script evaluates Q on the Shor code logical basis states:
    Q(|0_L⟩) = ⟨0_L|D̂|0_L⟩
    Q(|1_L⟩) = ⟨1_L|D̂|1_L⟩

Usage:
    python evaluate_Dhat_quadratic_form.py -i Dhat.txt

Arguments:
    -i, --input: Input file containing D̂ matrix (sparse format)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import simplify
from qef.matrix_io import read_sparse_matrix_symbolic, log
from qef.states import create_shor_logical_zero, create_shor_logical_one


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quadratic form Q(v) = ⟨v|D̂|v⟩ on code space basis"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file containing D̂ matrix'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        log(f"ERROR: File not found: {input_file}")
        return 1

    log("=" * 70)
    log(f"Evaluating quadratic form Q(v) = ⟨v|D̂|v⟩")
    log(f"Input file: {input_file}")
    log("=" * 70)
    log("")

    # Read D̂ matrix
    D_hat = read_sparse_matrix_symbolic(input_file)
    log("")

    # Create code space basis states
    log("Creating Shor code basis states...")
    ket_0L = create_shor_logical_zero()
    ket_1L = create_shor_logical_one()
    log("  |0_L⟩ created")
    log("  |1_L⟩ created")
    log("")

    # Evaluate Q(|0_L⟩) = ⟨0_L|D̂|0_L⟩
    log("Computing Q(|0_L⟩) = ⟨0_L|D̂|0_L⟩...")
    bra_0L = ket_0L.H
    Q_0L = (bra_0L * D_hat * ket_0L)[0, 0]
    log("  Simplifying...")
    Q_0L = simplify(Q_0L)
    log("")

    # Evaluate Q(|1_L⟩) = ⟨1_L|D̂|1_L⟩
    log("Computing Q(|1_L⟩) = ⟨1_L|D̂|1_L⟩...")
    bra_1L = ket_1L.H
    Q_1L = (bra_1L * D_hat * ket_1L)[0, 0]
    log("  Simplifying...")
    Q_1L = simplify(Q_1L)
    log("")

    # Report results
    log("=" * 70)
    log("RESULTS:")
    log("=" * 70)
    log("")
    log(f"Q(|0_L⟩) = {Q_0L}")
    log("")
    log(f"Q(|1_L⟩) = {Q_1L}")
    log("")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
