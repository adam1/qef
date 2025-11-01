"""
Error form computation for quantum error correction.

This module computes forms B_{λ,E,F} and D_{λ,E,F}.

B_{λ,E,F} = (E,F)*B - λ(E,F)·B

where (E,F)*B is the mixed pullback (E,F)*B(u,v) = B(Eu, Fv).
Matrix representation: B̂_{λ,E,F} = E†F - λ(E,F)·I

For error operators E, F and λ = λ(E,F), we compute:

D_{λ,E,F} = ∑_k conj(H_re(x,e_k))·H_re(e_k,y) + ∑_k conj(H_im(x,e_k))·H_im(e_k,y)

where:
- H_re = (1/2)(E+F)*B - Re(λ)B
- H_im = (1/2)(E+F)*B - Im(λ)iB

The pullback form G*B is defined by G*B(x,y) = B(Gx,Gy), which has
matrix representation G†G.

The matrix representation is D̂_{λ,E,F} = M_re†M_re + M_im†M_im where
M_re and M_im are the matrices for H_re and H_im respectively.

The total error form D̂ = ∑_{E,F} D̂_{λ,E,F} is computed by scripts that
parallelize over columns.
"""

from sympy import Matrix, re, im, I

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


def compute_D_lambda_EF(E_index: int, F_index: int, n_qubits: int, ket_v: Matrix) -> Matrix:
    """
    Compute the error form matrix D̂_{λ,E,F} for a pair of error operators.

    Given error operators E and F, and λ = λ(E,F), this computes the matrix
    for the sesquilinear form D_{λ,E,F} in the computational basis.

    The matrix is: D̂_{λ,E,F} = M_re†M_re + M_im†M_im

    where:
    - M_re = (1/2)(E+F)†(E+F) - Re(λ)I
    - M_im = (1/2)(E+F)†(E+F) - Im(λ)iI

    Args:
        E_index: Index for error operator E in the full Pauli basis
        F_index: Index for error operator F in the full Pauli basis
        n_qubits: Number of qubits
        ket_v: The quantum state |v⟩ as a column vector (Matrix)

    Returns:
        SymPy Matrix representing D̂_{λ,E,F} of dimension 2^n × 2^n
    """
    # Compute λ = λ(E, F)
    lambda_val = compute_lambda(E_index, F_index, n_qubits, ket_v)

    # Get Pauli operator matrices
    E = get_pauli_operator(E_index, n_qubits)
    F = get_pauli_operator(F_index, n_qubits)

    # Compute G = E + F
    G = E + F

    # Compute G†G (matrix for G*B)
    G_dag_G = G.H * G

    # Dimension of the Hilbert space
    dim = 2 ** n_qubits
    identity = Matrix.eye(dim)

    # Compute M_re = (1/2)G†G - Re(λ)I
    M_re = (G_dag_G / 2) - re(lambda_val) * identity

    # Compute M_im = (1/2)G†G - Im(λ)iI
    M_im = (G_dag_G / 2) - im(lambda_val) * I * identity

    # Compute D̂_{λ,E,F} = M_re†M_re + M_im†M_im
    # This ensures the result is Hermitian
    D_hat_lambda_EF = M_re.H * M_re + M_im.H * M_im

    return D_hat_lambda_EF
