#!/usr/bin/env python3
"""
Evaluate the quadratic form Q on all partitions of |0_L⟩.

The Shor code logical zero |0_L⟩ is a sum of 8 computational basis states.
This script partitions these 8 states into two non-empty subsets A and B,
creating vectors |a⟩ and |b⟩ such that |0_L⟩ = |a⟩ + |b⟩.

For each unordered partition, it evaluates:
    Q(|a⟩) = ⟨a|Ĵ|a⟩
    Q(|b⟩) = ⟨b|Ĵ|b⟩

There are (2^8 - 2)/2 = 127 unordered partitions.

Usage:
    python evaluate_Jhat_on_0L_partitions.py -i Jhat.txt -o output_dir

Arguments:
    -i, --input: Input file containing Ĵ matrix (sparse format)
    -o, --output: Output directory for results
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations
from datetime import datetime
import os

# Add parent directory to path to import qef
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sympy import Matrix, simplify, sqrt
from qef.matrix_io import read_sparse_matrix_symbolic, log


def timestamp():
    """Return current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_id():
    """Generate a run ID (timestamp + PID)."""
    return f"{int(datetime.now().timestamp())}_{os.getpid()}"


def get_0L_basis_states():
    """
    Get the 8 computational basis states that appear in |0_L⟩.

    For the 9-qubit Shor code:
    |0_L⟩ = (1/√8) * Σ of 8 basis states

    The states are (|000⟩ + |111⟩)^⊗3, which expands to:
    |000000000⟩, |000000111⟩, |000111000⟩, |000111111⟩,
    |111000000⟩, |111000111⟩, |111111000⟩, |111111111⟩

    Returns:
        List of 8 integers (basis state indices)
    """
    # Convert binary strings to integers
    states = [
        0b000000000,  # 0
        0b000000111,  # 7
        0b000111000,  # 56
        0b000111111,  # 63
        0b111000000,  # 448
        0b111000111,  # 455
        0b111111000,  # 504
        0b111111111,  # 511
    ]
    return states


def create_vector_from_basis_states(basis_states, n_qubits=9):
    """
    Create a state vector from a set of computational basis states.

    Args:
        basis_states: List of basis state indices
        n_qubits: Number of qubits (default: 9)

    Returns:
        SymPy Matrix (column vector) with 1/√k at specified positions
    """
    dim = 2 ** n_qubits
    k = len(basis_states)

    vec = Matrix.zeros(dim, 1)
    amplitude = 1 / sqrt(k)

    for state_idx in basis_states:
        vec[state_idx] = amplitude

    return vec


def partition_index_to_subset(index, n_elements):
    """
    Convert a partition index to a subset.

    We enumerate all non-empty proper subsets (excluding empty set and full set).
    Index i corresponds to the binary representation where bit j indicates
    whether element j is in the subset.

    Args:
        index: Partition index (1 to 2^n - 2)
        n_elements: Total number of elements

    Returns:
        Set of element indices in the subset
    """
    subset = set()
    for j in range(n_elements):
        if index & (1 << j):
            subset.add(j)
    return subset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Q(|a⟩) and Q(|b⟩) for all partitions of |0_L⟩"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file containing Ĵ matrix'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for results'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        log(f"ERROR: File not found: {input_file}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    run_tag = run_id()

    log("=" * 70)
    log(f"Evaluating Q on all partitions of |0_L⟩")
    log(f"Input file: {input_file}")
    log(f"Output directory: {output_dir}")
    log(f"Run ID: {run_tag}")
    log("=" * 70)
    log("")

    # Read Ĵ matrix
    J_hat = read_sparse_matrix_symbolic(input_file)
    log("")

    # Get the 8 basis states in |0_L⟩
    basis_states = get_0L_basis_states()
    log(f"Basis states in |0_L⟩: {len(basis_states)}")
    log(f"States: {[bin(s) for s in basis_states]}")
    log("")

    # Calculate number of partitions
    n_partitions = (2**len(basis_states) - 2) // 2
    log(f"Number of unordered partitions: {n_partitions}")
    log("")

    # Output file
    output_file = output_dir / f"Jhat_0L_partitions_{run_tag}.txt"

    log(f"Writing results to: {output_file}")
    log("")

    partition_count = 0

    with open(output_file, 'w') as f:
        f.write(f"# Quadratic form Q(v) = ⟨v|Ĵ|v⟩ evaluated on partitions of |0_L⟩\n")
        f.write(f"# Run ID: {run_tag}\n")
        f.write(f"# Timestamp: {timestamp()}\n")
        f.write(f"# Total partitions: {n_partitions}\n")
        f.write(f"#\n")
        f.write(f"# Format: partition_id |A| |B| Q(|a⟩) Q(|b⟩) A_indices B_indices\n")
        f.write("\n")

        # Iterate over all non-empty proper subsets (partition representatives)
        # We only iterate up to the "first half" to avoid double-counting
        for subset_bits in range(1, 2**len(basis_states) - 1):
            # To avoid counting both {A, B} and {B, A}, only count when A is "smaller"
            # (e.g., when the smallest element is in A)
            complement_bits = (2**len(basis_states) - 1) ^ subset_bits

            # Only process if subset_bits < complement_bits (lexicographically smaller)
            if subset_bits >= complement_bits:
                continue

            partition_count += 1

            # Convert bit patterns to subsets of basis state indices
            A_indices = partition_index_to_subset(subset_bits, len(basis_states))
            B_indices = partition_index_to_subset(complement_bits, len(basis_states))

            # Get actual basis states
            A_states = [basis_states[i] for i in sorted(A_indices)]
            B_states = [basis_states[i] for i in sorted(B_indices)]

            if partition_count % 10 == 0:
                log(f"[{partition_count}/{n_partitions}] Processing partition {partition_count}...")

            # Create vectors |a⟩ and |b⟩
            ket_a = create_vector_from_basis_states(A_states)
            ket_b = create_vector_from_basis_states(B_states)

            # Compute Q(|a⟩) = ⟨a|Ĵ|a⟩
            bra_a = ket_a.H
            Q_a = (bra_a * J_hat * ket_a)[0, 0]
            Q_a = simplify(Q_a)

            # Compute Q(|b⟩) = ⟨b|Ĵ|b⟩
            bra_b = ket_b.H
            Q_b = (bra_b * J_hat * ket_b)[0, 0]
            Q_b = simplify(Q_b)

            # Write results
            A_idx_str = ','.join(map(str, sorted(A_indices)))
            B_idx_str = ','.join(map(str, sorted(B_indices)))

            f.write(f"{partition_count} {len(A_indices)} {len(B_indices)} {Q_a} {Q_b} [{A_idx_str}] [{B_idx_str}]\n")
            f.flush()

    log("")
    log("=" * 70)
    log(f"Processed {partition_count} partitions")
    log(f"Results written to: {output_file}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
