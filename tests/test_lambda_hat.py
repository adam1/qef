"""Unit tests for lambda_hat module (lambda-hat computation)."""

import pytest
from sympy import Matrix, sqrt, simplify, conjugate, I

from qef.lambda_hat import compute_lambda
from qef.states import create_shor_logical_zero
from qef.operators import get_basis_P_n_t, get_pauli_operator, pauli_string_to_index


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

    def test_lambda_shor_code(self):
        """Test λ values for Shor code with Hermitian property spot checks."""
        ket_v = create_shor_logical_zero()
        n_qubits = 9

        # Basic checks
        # λ(I, I) = ⟨0_L|I†I|0_L⟩ = 1
        entry = compute_lambda(0, 0, n_qubits, ket_v)
        assert simplify(entry) == 1

        # λ(I, X_0) should be 0 for Shor code
        entry = compute_lambda(0, 1, n_qubits, ket_v)
        assert simplify(entry) == 0

        # λ(Z_1, Z_2) should be 1
        entry = compute_lambda(
            pauli_string_to_index("ZII III III", 9),
            pauli_string_to_index("IZI III III", 9),
            n_qubits, ket_v)
        assert simplify(entry) == 1

        # Hermitian property spot checks: λ(s, t) = conj(λ(t, s))

        # Check (I, X_0) vs (X_0, I)
        lambda_I_X0 = compute_lambda(0, 1, n_qubits, ket_v)
        lambda_X0_I = compute_lambda(1, 0, n_qubits, ket_v)
        assert simplify(lambda_I_X0 - conjugate(lambda_X0_I)) == 0

        # Check (I, Y_0) vs (Y_0, I)
        lambda_I_Y0 = compute_lambda(0, 2, n_qubits, ket_v)
        lambda_Y0_I = compute_lambda(2, 0, n_qubits, ket_v)
        assert simplify(lambda_I_Y0 - conjugate(lambda_Y0_I)) == 0

        # Check (I, Z_0) vs (Z_0, I)
        lambda_I_Z0 = compute_lambda(0, 3, n_qubits, ket_v)
        lambda_Z0_I = compute_lambda(3, 0, n_qubits, ket_v)
        assert simplify(lambda_I_Z0 - conjugate(lambda_Z0_I)) == 0

        # Check (X_0, Y_0) vs (Y_0, X_0)
        lambda_X0_Y0 = compute_lambda(1, 2, n_qubits, ket_v)
        lambda_Y0_X0 = compute_lambda(2, 1, n_qubits, ket_v)
        assert simplify(lambda_X0_Y0 - conjugate(lambda_Y0_X0)) == 0

        # Check (X_0, Z_0) vs (Z_0, X_0)
        lambda_X0_Z0 = compute_lambda(1, 3, n_qubits, ket_v)
        lambda_Z0_X0 = compute_lambda(3, 1, n_qubits, ket_v)
        assert simplify(lambda_X0_Z0 - conjugate(lambda_Z0_X0)) == 0

        # Check (Y_0, Z_0) vs (Z_0, Y_0)
        lambda_Y0_Z0 = compute_lambda(2, 3, n_qubits, ket_v)
        lambda_Z0_Y0 = compute_lambda(3, 2, n_qubits, ket_v)
        assert simplify(lambda_Y0_Z0 - conjugate(lambda_Z0_Y0)) == 0

        # Check across different qubits: (X_0, X_1) vs (X_1, X_0)
        # X_1 has index 1 + 3*1 = 4
        lambda_X0_X1 = compute_lambda(1, 4, n_qubits, ket_v)
        lambda_X1_X0 = compute_lambda(4, 1, n_qubits, ket_v)
        assert simplify(lambda_X0_X1 - conjugate(lambda_X1_X0)) == 0

        # Check (Z_1, Z_2) vs (Z_2, Z_1)
        lambda_Y0_Z0 = compute_lambda(3, 7, n_qubits, ket_v)
        lambda_Z0_Y0 = compute_lambda(7, 3, n_qubits, ket_v)
        assert simplify(lambda_Y0_Z0 - conjugate(lambda_Z0_Y0)) == 0

