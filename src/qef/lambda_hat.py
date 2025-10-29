"""
Hermitian form computation for quantum error correction.

This module computes the Hermitian form λ : E × E → C defined by
λ(E, F) = ⟨v|E†F|v⟩.

The matrix representation λ̂ is computed by scripts that call compute_lambda
for each entry.
"""

from sympy import Matrix

from .operators import get_pauli_operator


def compute_lambda(s_index: int, t_index: int, n_qubits: int, ket_v: Matrix) -> complex:
    """
    Compute the value of the Hermitian form λ(E_s, E_t).

    λ(E_s, E_t) = ⟨v|E_s† E_t|v⟩

    This value becomes the (s,t) entry of the matrix λ̂.

    Args:
        s_index: Index for error operator E_s in the full Pauli basis
        t_index: Index for error operator E_t in the full Pauli basis
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector (Matrix)

    Returns:
        Complex number (SymPy expression)
    """
    # Get Pauli operators
    E_s = get_pauli_operator(s_index, n_qubits)
    E_t = get_pauli_operator(t_index, n_qubits)

    # Compute E_s† E_t |v⟩
    E_s_dag = E_s.H  # Hermitian conjugate

    # Compute E_s† E_t |v⟩
    result_ket = E_s_dag * E_t * ket_v

    # Compute ⟨v| (result_ket) = (ket_v)† * result_ket
    bra_v = ket_v.H

    # Inner product
    entry = (bra_v * result_ket)[0, 0]  # Extract scalar from 1x1 matrix

    return entry
