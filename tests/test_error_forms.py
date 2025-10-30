"""Unit tests for error_forms module."""

import pytest
from sympy import Matrix, sqrt, simplify, conjugate, I, re, im

from qef.error_forms import compute_D_lambda_EF
from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t


class TestDLambdaEF:
    """Test D_{λ,E,F} computation."""

    def test_D_lambda_EF_dimension(self):
        """Test that D̂_{λ,E,F} has correct dimension."""
        # Use simple 2-qubit state
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2

        # Compute D̂_{λ,I,I}
        D = compute_D_lambda_EF(0, 0, n_qubits, ket_v)

        # Should be 4×4 matrix for 2 qubits
        assert D.shape == (4, 4)

    def test_D_lambda_EF_hermitian(self):
        """Test that D̂_{λ,E,F} is Hermitian."""
        # Use simple 2-qubit state
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2

        # Test a few combinations
        test_pairs = [(0, 0), (0, 1), (1, 2)]

        for E_idx, F_idx in test_pairs:
            D = compute_D_lambda_EF(E_idx, F_idx, n_qubits, ket_v)

            # Check Hermitian property: D = D†
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    assert simplify(D[i, j] - conjugate(D[j, i])) == 0

    def test_D_lambda_II_zero_state(self):
        """Test D̂_{λ,I,I} for |00⟩ state."""
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2

        # For E = F = I:
        # λ(I,I) = ⟨00|I|00⟩ = 1 (real)
        # G = I + I = 2I
        # G†G = 4I
        # M_re = (1/2)·4I - 1·I = 2I - I = I
        # M_im = (1/2)·4I - 0·iI = 2I
        # D̂ = M_re†M_re + M_im†M_im = I†I + (2I)†(2I) = I + 4I = 5I

        D = compute_D_lambda_EF(0, 0, n_qubits, ket_v)

        # Should be 5I (5 times identity)
        expected = 5 * Matrix.eye(4)

        for i in range(4):
            for j in range(4):
                assert simplify(D[i, j] - expected[i, j]) == 0

    def test_D_lambda_EF_shor_code(self):
        """Test D̂_{λ,E,F} for Shor code (spot check)."""
        ket_v = create_shor_logical_zero()
        n_qubits = 9

        # Compute D̂_{λ,I,I}
        D = compute_D_lambda_EF(0, 0, n_qubits, ket_v)

        # Should be 512×512 matrix for 9 qubits
        assert D.shape == (512, 512)

        # Should be Hermitian
        # Just check diagonal is real (full check would be expensive)
        for i in range(10):  # Check first 10 diagonal elements
            entry = D[i, i]
            assert simplify(entry - conjugate(entry)) == 0
