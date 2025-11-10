"""
Matrix utility functions for quantum error correction.

This module provides utilities for checking properties of matrices
used in quantum error correction computations.
"""

from sympy import Matrix, simplify
import numpy as np


def is_hermitian(M: Matrix) -> bool:
    """
    Check if a matrix is Hermitian (symbolically).

    A matrix M is Hermitian if M = M†, where M† is the conjugate transpose.

    Args:
        M: A SymPy Matrix to check

    Returns:
        True if M is Hermitian, False otherwise

    Examples:
        >>> from sympy import Matrix, I
        >>> M = Matrix([[1, I], [-I, 1]])
        >>> is_hermitian(M)
        True

        >>> M = Matrix([[1, 1], [0, 1]])
        >>> is_hermitian(M)
        False
    """
    if M.rows != M.cols:
        return False

    # Compute conjugate transpose
    M_dagger = M.H

    # Check if M = M† by checking if M - M† = 0
    diff = M - M_dagger

    # Symbolically simplify and check each entry
    for i in range(diff.rows):
        for j in range(diff.cols):
            if simplify(diff[i, j]) != 0:
                return False

    return True


def is_skew_hermitian(M: Matrix) -> bool:
    """
    Check if a matrix is skew-Hermitian (symbolically).

    A matrix M is skew-Hermitian if M† = -M, where M† is the conjugate transpose.
    Equivalently, M is skew-Hermitian if M + M† = 0.

    Args:
        M: A SymPy Matrix to check

    Returns:
        True if M is skew-Hermitian, False otherwise

    Examples:
        >>> from sympy import Matrix, I
        >>> M = Matrix([[0, 1+I], [-(1-I), 0]])
        >>> is_skew_hermitian(M)
        True

        >>> M = Matrix([[1, I], [-I, 1]])
        >>> is_skew_hermitian(M)
        False
    """
    if M.rows != M.cols:
        return False

    # Compute conjugate transpose
    M_dagger = M.H

    # Check if M† = -M by checking if M + M† = 0
    sum_matrix = M + M_dagger

    # Symbolically simplify and check each entry
    for i in range(sum_matrix.rows):
        for j in range(sum_matrix.cols):
            if simplify(sum_matrix[i, j]) != 0:
                return False

    return True


def sympy_to_numpy(matrix: Matrix) -> np.ndarray:
    """
    Convert SymPy Matrix to NumPy array with complex float values.

    Uses SymPy's applyfunc with evalf() to numerically evaluate each element,
    then converts to a NumPy array with complex dtype.

    Args:
        matrix: SymPy Matrix to convert

    Returns:
        NumPy array with dtype=complex

    Examples:
        >>> from sympy import Matrix, sqrt, I
        >>> M = Matrix([[1, sqrt(2)], [I, 1+I]])
        >>> M_np = sympy_to_numpy(M)
        >>> M_np.dtype
        dtype('complex128')
    """
    return np.array(matrix.applyfunc(lambda x: complex(x.evalf())).tolist(), dtype=complex)


def compute_signature(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> tuple:
    """
    Compute signature (p, q, r) and Witt index from eigenvalues.

    For a Hermitian matrix with given eigenvalues, classifies them
    as positive, negative, or zero, and computes the Witt index.

    Args:
        eigenvalues: Array of eigenvalues (real numbers)
        tolerance: Threshold for considering eigenvalues as zero

    Returns:
        Tuple of (p, q, r, witt_index) where:
          p = number of positive eigenvalues
          q = number of negative eigenvalues
          r = nullity (number of zero eigenvalues)
          witt_index = min(p, q) (dimension of maximal totally isotropic subspace)

    Examples:
        >>> eigenvalues = np.array([2, 1, 0, -1, -2])
        >>> p, q, r, witt = compute_signature(eigenvalues)
        >>> (p, q, r, witt)
        (2, 2, 1, 2)
    """
    p = int(np.sum(eigenvalues > tolerance))
    q = int(np.sum(eigenvalues < -tolerance))
    r = int(np.sum(np.abs(eigenvalues) <= tolerance))
    witt_index = min(p, q)

    return p, q, r, witt_index
