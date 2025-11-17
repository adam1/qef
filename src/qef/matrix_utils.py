"""
Matrix utility functions for quantum error correction.

This module provides utilities for checking properties of matrices
used in quantum error correction computations.
"""

from sympy import Matrix, simplify


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
