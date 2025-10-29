"""
Lambda-hat matrix computation for quantum error correction.

This module computes the matrix representation λ̂ (lambda-hat) of the
Hermitian form λ : E × E → C defined by λ(E, F) = ⟨v|E†F|v⟩.
"""

from sympy import Matrix, conjugate
from typing import List

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


def compute_lambda_hat(basis_indices: List[int], n_qubits: int, ket_v: Matrix,
                       verbose: bool = False) -> Matrix:
    """
    Compute the matrix λ̂ representing the Hermitian form λ in a given basis.

    Since λ̂ is Hermitian, we only compute the upper triangular part
    and fill the lower part using λ̂_{i,j} = conj(λ̂_{j,i}).

    Args:
        basis_indices: List of indices in the full Pauli basis (e.g., from get_basis_P_n_t)
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector
        verbose: If True, print progress for each entry

    Returns:
        SymPy Matrix λ̂ of dimension len(basis_indices) × len(basis_indices)
    """
    matrix_dim = len(basis_indices)

    # Initialize the matrix λ̂
    lambda_hat = Matrix.zeros(matrix_dim, matrix_dim)

    # Compute upper triangular part (including diagonal)
    for i in range(matrix_dim):
        for j in range(i, matrix_dim):  # Only j >= i (upper triangular)
            s_index = basis_indices[i]
            t_index = basis_indices[j]

            if verbose:
                print(f"Computing λ̂[{i},{j}]: basis indices ({s_index}, {t_index})")

            entry = compute_lambda(s_index, t_index, n_qubits, ket_v)
            lambda_hat[i, j] = entry.simplify()

    # Fill lower triangular part using Hermitian property
    for i in range(matrix_dim):
        for j in range(i):  # Only j < i (strictly lower triangular)
            lambda_hat[i, j] = conjugate(lambda_hat[j, i])

    return lambda_hat
