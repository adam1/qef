"""Tests for matrix utility functions."""

import pytest
from sympy import Matrix, I, symbols

from qef.matrix_utils import is_hermitian


def test_is_hermitian_identity():
    """Test that identity matrix is Hermitian."""
    M = Matrix.eye(3)
    assert is_hermitian(M)


def test_is_hermitian_diagonal_real():
    """Test that diagonal matrix with real entries is Hermitian."""
    M = Matrix([[1, 0, 0],
                [0, 2, 0],
                [0, 0, 3]])
    assert is_hermitian(M)


def test_is_hermitian_with_complex():
    """Test Hermitian matrix with complex off-diagonal entries."""
    M = Matrix([[1, I, 0],
                [-I, 2, 1 + I],
                [0, 1 - I, 3]])
    assert is_hermitian(M)


def test_not_hermitian_complex():
    """Test that non-Hermitian matrix with complex entries is detected."""
    M = Matrix([[1, I],
                [I, 1]])  # Should be -I in (1,0) position for Hermitian
    assert not is_hermitian(M)


def test_not_hermitian_asymmetric():
    """Test that asymmetric matrix is not Hermitian."""
    M = Matrix([[1, 2],
                [3, 4]])
    assert not is_hermitian(M)


def test_is_hermitian_symbolic():
    """Test Hermitian matrix with symbolic entries."""
    a, b = symbols('a b', real=True)
    M = Matrix([[a, I],
                [-I, b]])
    assert is_hermitian(M)


def test_not_hermitian_symbolic():
    """Test non-Hermitian symbolic matrix."""
    a, b = symbols('a b')
    M = Matrix([[a, b],
                [0, a]])
    assert not is_hermitian(M)


def test_is_hermitian_non_square():
    """Test that non-square matrix is not Hermitian."""
    M = Matrix([[1, 2, 3],
                [4, 5, 6]])
    assert not is_hermitian(M)


def test_is_hermitian_pauli_x():
    """Test that Pauli X matrix is Hermitian."""
    X = Matrix([[0, 1],
                [1, 0]])
    assert is_hermitian(X)


def test_is_hermitian_pauli_y():
    """Test that Pauli Y matrix is Hermitian."""
    Y = Matrix([[0, -I],
                [I, 0]])
    assert is_hermitian(Y)


def test_is_hermitian_pauli_z():
    """Test that Pauli Z matrix is Hermitian."""
    Z = Matrix([[1, 0],
                [0, -1]])
    assert is_hermitian(Z)
