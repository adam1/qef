"""
Error form computation for quantum error correction.

This module computes the form B_{λ,E,F}.

B_{λ,E,F} = (E,F)*B - λ(E,F)·B

where (E,F)*B is the mixed pullback (E,F)*B(u,v) = B(Eu, Fv).
Matrix representation: B̂_{λ,E,F} = E†F - λ(E,F)·I
"""

from sympy import Matrix

from .operators import get_pauli_operator
from .lambda_hat import compute_lambda


def compute_B_hat_lambda_EF(E_index: int, F_index: int, n_qubits: int, ket_v: Matrix) -> Matrix:
    """
    Compute the form matrix B̂_{λ,E,F} for a pair of error operators.

    Given error operators E and F, and λ = λ(E,F), this computes the matrix
    for the form B_{λ,E,F} in the computational basis.

    The form is defined as:
        B_{λ,E,F} = (E,F)*B - λ(E,F)·B

    where (E,F)*B is the mixed pullback defined by (E,F)*B(u,v) = B(Eu, Fv).

    The matrix representation is:
        B̂_{λ,E,F} = E†F - λ(E,F)·I

    Args:
        E_index: Index for error operator E in the full Pauli basis
        F_index: Index for error operator F in the full Pauli basis
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector (Matrix)

    Returns:
        SymPy Matrix representing B̂_{λ,E,F} of dimension 2^n × 2^n
    """
    # Compute λ = λ(E, F)
    lambda_val = compute_lambda(E_index, F_index, n_qubits, ket_v)

    # Get Pauli operator matrices
    E = get_pauli_operator(E_index, n_qubits)
    F = get_pauli_operator(F_index, n_qubits)

    # Dimension of the Hilbert space
    dim = 2 ** n_qubits
    identity = Matrix.eye(dim)

    # Compute B̂_{λ,E,F} = E†F - λ(E,F)·I
    B_hat_lambda_EF = E.H * F - lambda_val * identity

    return B_hat_lambda_EF
