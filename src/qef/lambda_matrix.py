"""
Lambda matrix computation for quantum error correction.
"""

from sympy import Matrix, conjugate
from typing import List

from .operators import get_pauli_operator


def compute_lambda_entry(s_index: int, t_index: int, n_qubits: int, ket_v: Matrix) -> complex:
    """
    Compute a single entry of the Hermitian form matrix λ.

    λ_{s,t} = ⟨v|P_s† P_t|v⟩

    Args:
        s_index: Row index (full basis index for P_s)
        t_index: Column index (full basis index for P_t)
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector (Matrix)

    Returns:
        Complex number (SymPy expression)
    """
    # Get Pauli operators
    P_s = get_pauli_operator(s_index, n_qubits)
    P_t = get_pauli_operator(t_index, n_qubits)

    # Compute P_s† P_t |v⟩
    P_s_dag = P_s.H  # Hermitian conjugate

    # Compute P_s† P_t |v⟩
    result_ket = P_s_dag * P_t * ket_v

    # Compute ⟨v| (result_ket) = (ket_v)† * result_ket
    bra_v = ket_v.H

    # Inner product
    entry = (bra_v * result_ket)[0, 0]  # Extract scalar from 1x1 matrix

    return entry


def compute_lambda_matrix(basis_indices: List[int], n_qubits: int, ket_v: Matrix,
                         verbose: bool = False) -> Matrix:
    """
    Compute the Hermitian form matrix λ for a given basis.

    Since λ is Hermitian, we only compute the upper triangular part
    and fill the lower part using λ_{i,j} = conj(λ_{j,i}).

    Args:
        basis_indices: List of indices in the full Pauli basis (e.g., from get_basis_P_n_t)
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector
        verbose: If True, print progress for each entry

    Returns:
        SymPy Matrix of dimension len(basis_indices) × len(basis_indices)
    """
    matrix_dim = len(basis_indices)

    # Initialize the matrix
    lambda_matrix = Matrix.zeros(matrix_dim, matrix_dim)

    # Compute upper triangular part (including diagonal)
    for i in range(matrix_dim):
        for j in range(i, matrix_dim):  # Only j >= i (upper triangular)
            s_index = basis_indices[i]
            t_index = basis_indices[j]

            if verbose:
                print(f"Computing λ[{i},{j}]: basis indices ({s_index}, {t_index})")

            entry = compute_lambda_entry(s_index, t_index, n_qubits, ket_v)
            lambda_matrix[i, j] = entry.simplify()

    # Fill lower triangular part using Hermitian property
    for i in range(matrix_dim):
        for j in range(i):  # Only j < i (strictly lower triangular)
            lambda_matrix[i, j] = conjugate(lambda_matrix[j, i])

    return lambda_matrix
