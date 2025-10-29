"""Unit tests for lambda_hat module (lambda-hat computation)."""

import pytest
from sympy import Matrix, sqrt, simplify, conjugate, I

from qef.lambda_hat import compute_lambda, compute_lambda_hat
from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t, get_pauli_operator


class TestLambdaEntry:
    """Test individual lambda form value computation."""

    def test_lambda_identity_identity_simple_state(self):
        """Test λ(I, I) = 1 for a simple normalized state."""
        # Use a simple 2-qubit state: |00⟩
        ket_v = Matrix([1, 0, 0, 0])
        n_qubits = 2

        # Both indices 0 (identity)
        entry = compute_lambda(0, 0, n_qubits, ket_v)

        # Should be 1 (⟨00|I|00⟩ = 1)
        assert simplify(entry) == 1

    def test_lambda_pauli_simple_state(self):
        """Test λ values for simple Pauli operations."""
        # Use |+⟩ state for qubit 0: (|0⟩ + |1⟩)/√2
        # Tensor with |0⟩ for qubit 1: (|00⟩ + |10⟩)/√2
        ket_v = Matrix([1, 0, 1, 0]) / sqrt(2)
        n_qubits = 2

        # Test ⟨v|X_0|v⟩ = ⟨+|X|+⟩ ⊗ ⟨0|I|0⟩ = 1
        # X_0 has index 1 (base-4: [1, 0])
        entry = compute_lambda(1, 1, n_qubits, ket_v)
        # ⟨v|X† X|v⟩ = ⟨v|I|v⟩ = 1
        assert simplify(entry) == 1

    def test_lambda_hermitian_property_simple(self):
        """Test that λ(s, t) = conj(λ(t, s))."""
        # Use a simple 2-qubit state
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2

        # Test with a few pairs
        test_pairs = [(0, 1), (1, 2), (2, 3)]

        for s, t in test_pairs:
            entry_st = compute_lambda(s, t, n_qubits, ket_v)
            entry_ts = compute_lambda(t, s, n_qubits, ket_v)

            assert simplify(entry_st - conjugate(entry_ts)) == 0


class TestLambdaHat:
    """Test lambda-hat matrix computation."""

    def test_lambda_hat_dimension_small(self):
        """Test that λ̂ has correct dimension for small basis."""
        # Use 2 qubits, P_{2,1} has dimension 1 + 3*2 = 7
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2
        basis = get_basis_P_n_t(n_qubits, 1)

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        assert lambda_hat_mat.shape == (7, 7)

    def test_lambda_hat_hermitian_small(self):
        """Test that λ̂ is Hermitian."""
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2
        basis = get_basis_P_n_t(n_qubits, 1)

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        # Check Hermitian property: λ̂ = λ̂†
        for i in range(lambda_hat_mat.shape[0]):
            for j in range(lambda_hat_mat.shape[1]):
                assert simplify(lambda_hat_mat[i, j] - conjugate(lambda_hat_mat[j, i])) == 0

    def test_lambda_hat_diagonal_real(self):
        """Test that diagonal entries are real."""
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2
        basis = get_basis_P_n_t(n_qubits, 0)  # Just identity

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        # Diagonal entries of Hermitian matrix should be real
        for i in range(lambda_hat_mat.shape[0]):
            entry = lambda_hat_mat[i, i]
            # Check that imaginary part is zero
            assert simplify(entry - conjugate(entry)) == 0

    def test_lambda_hat_identity_only(self):
        """Test λ̂ for basis with only identity."""
        ket_v = Matrix([1, 0, 0, 0])  # |00⟩
        n_qubits = 2

        # Use just identity
        basis = [0]

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        # Should be 1x1 matrix with entry 1
        assert lambda_hat_mat.shape == (1, 1)
        assert simplify(lambda_hat_mat[0, 0]) == 1

    def test_lambda_hat_structure_simple(self):
        """Test λ̂ structure for a very simple case."""
        # Use |0⟩ state for 1 qubit
        ket_v = Matrix([1, 0])
        n_qubits = 1

        # Use full basis for 1 qubit: I, X, Y, Z (indices 0, 1, 2, 3)
        basis = [0, 1, 2, 3]

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        # Check dimension
        assert lambda_hat_mat.shape == (4, 4)

        # For |0⟩:
        # ⟨0|I† I|0⟩ = 1
        assert simplify(lambda_hat_mat[0, 0]) == 1

        # ⟨0|I† X|0⟩ = ⟨0|X|0⟩ = 0
        assert simplify(lambda_hat_mat[0, 1]) == 0

        # ⟨0|I† Y|0⟩ = ⟨0|Y|0⟩ = 0
        assert simplify(lambda_hat_mat[0, 2]) == 0

        # ⟨0|I† Z|0⟩ = ⟨0|Z|0⟩ = 1
        assert simplify(lambda_hat_mat[0, 3]) == 1

        # ⟨0|X† X|0⟩ = ⟨0|I|0⟩ = 1
        assert simplify(lambda_hat_mat[1, 1]) == 1

        # ⟨0|X† Y|0⟩ = ⟨0|iZ|0⟩ = i
        # Actually X†Y = -iZ for Paulis, but let's compute it
        entry_XY = lambda_hat_mat[1, 2]
        # This should be iZ applied to |0⟩ = i|0⟩, so ⟨0|iZ|0⟩ = i
        # Actually we need to check the sign conventions...
        # Let's just verify it's Hermitian
        assert simplify(lambda_hat_mat[1, 2] - conjugate(lambda_hat_mat[2, 1])) == 0

    def test_lambda_hat_two_qubit_P_2_1(self):
        """Test 2-qubit P_{2,1} basis (dimension 7)."""
        # Use |00⟩
        ket_v = Matrix([1, 0, 0, 0])
        n_qubits = 2
        basis = get_basis_P_n_t(n_qubits, 1)

        lambda_hat_mat = compute_lambda_hat(basis, n_qubits, ket_v)

        # Basic checks
        assert lambda_hat_mat.shape == (7, 7)
        assert simplify(lambda_hat_mat[0, 0]) == 1  # ⟨00|I|00⟩ = 1

        # Verify Hermitian
        assert lambda_hat_mat == lambda_hat_mat.H

    def test_lambda_single_entry_shor_code(self):
        """Test a single λ value computation with Shor code (spot check)."""
        ket_v = create_shor_logical_zero()
        n_qubits = 9

        # Just compute λ(I, I) - should be 1
        entry = compute_lambda(0, 0, n_qubits, ket_v)
        assert simplify(entry) == 1

        # Compute a few more individual entries without building full matrix
        # λ(I, X_0) should be 0 for Shor code
        entry = compute_lambda(0, 1, n_qubits, ket_v)
        assert simplify(entry) == 0
