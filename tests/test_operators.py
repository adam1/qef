"""Unit tests for operators module."""

import pytest
from sympy import Matrix, I
from sympy.physics.matrices import msigma

from qef.operators import (
    index_to_base4,
    base4_to_pauli_string,
    index_to_pauli_string,
    pauli_string_to_index,
    get_pauli_operator,
    count_nontrivial_qubits,
    get_basis_P_n_t,
)


class TestBase4Indexing:
    """Test base-4 indexing conversions."""

    def test_index_to_base4_identity(self):
        """Test that index 0 gives all zeros."""
        assert index_to_base4(0, 9) == [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_index_to_base4_examples_from_readme(self):
        """Test the specific examples from README.md."""
        # Index 1 → 000 000 001
        assert index_to_base4(1, 9) == [1, 0, 0, 0, 0, 0, 0, 0, 0]

        # Index 5 = 1*4^0 + 1*4^1 → 000 000 011
        assert index_to_base4(5, 9) == [1, 1, 0, 0, 0, 0, 0, 0, 0]

        # Index 9 = 1*4^0 + 2*4^1 → 000 000 021
        assert index_to_base4(9, 9) == [1, 2, 0, 0, 0, 0, 0, 0, 0]

        # Index 262143 = 4^9 - 1 → 333 333 333
        assert index_to_base4(262143, 9) == [3, 3, 3, 3, 3, 3, 3, 3, 3]

    def test_index_to_base4_small_n(self):
        """Test with smaller number of qubits."""
        assert index_to_base4(0, 2) == [0, 0]
        assert index_to_base4(1, 2) == [1, 0]
        assert index_to_base4(2, 2) == [2, 0]
        assert index_to_base4(3, 2) == [3, 0]
        assert index_to_base4(4, 2) == [0, 1]
        assert index_to_base4(5, 2) == [1, 1]

    def test_index_to_base4_out_of_range(self):
        """Test that out-of-range indices raise errors."""
        with pytest.raises(ValueError):
            index_to_base4(-1, 9)
        with pytest.raises(ValueError):
            index_to_base4(4**9, 9)


class TestPauliStrings:
    """Test Pauli string conversions."""

    def test_base4_to_pauli_string_identity(self):
        """Test identity operator."""
        assert base4_to_pauli_string([0, 0, 0, 0, 0, 0, 0, 0, 0]) == "III III III"

    def test_base4_to_pauli_string_readme_examples(self):
        """Test examples from README."""
        # Index 1 → XII III III
        assert base4_to_pauli_string([1, 0, 0, 0, 0, 0, 0, 0, 0]) == "XII III III"

        # Index 5 → XXI III III
        assert base4_to_pauli_string([1, 1, 0, 0, 0, 0, 0, 0, 0]) == "XXI III III"

        # Index 9 → XYI III III
        assert base4_to_pauli_string([1, 2, 0, 0, 0, 0, 0, 0, 0]) == "XYI III III"

        # Index 262143 → ZZZ ZZZ ZZZ
        assert base4_to_pauli_string([3, 3, 3, 3, 3, 3, 3, 3, 3]) == "ZZZ ZZZ ZZZ"

    def test_index_to_pauli_string_readme_examples(self):
        """Test direct index to string conversion."""
        assert index_to_pauli_string(0, 9) == "III III III"
        assert index_to_pauli_string(1, 9) == "XII III III"
        assert index_to_pauli_string(5, 9) == "XXI III III"
        assert index_to_pauli_string(9, 9) == "XYI III III"
        assert index_to_pauli_string(262143, 9) == "ZZZ ZZZ ZZZ"

    def test_pauli_string_to_index_roundtrip(self):
        """Test that string → index → string is identity."""
        test_indices = [0, 1, 5, 9, 100, 262143]
        for idx in test_indices:
            pauli_str = index_to_pauli_string(idx, 9)
            recovered_idx = pauli_string_to_index(pauli_str, 9)
            assert recovered_idx == idx

    def test_pauli_string_to_index_with_spaces(self):
        """Test that spaces are handled correctly."""
        assert pauli_string_to_index("XII III III", 9) == 1
        assert pauli_string_to_index("XIIIIIIII", 9) == 1

    def test_pauli_string_to_index_invalid(self):
        """Test error handling for invalid strings."""
        with pytest.raises(ValueError):
            pauli_string_to_index("XII III", 9)  # Too short
        with pytest.raises(ValueError):
            pauli_string_to_index("QIIIIIII", 9)  # Invalid character


class TestPauliOperators:
    """Test Pauli operator matrix construction."""

    def test_get_pauli_operator_identity(self):
        """Test that index 0 gives identity."""
        n = 2
        P = get_pauli_operator(0, n)
        expected = Matrix.eye(2**n)
        assert P == expected

    def test_get_pauli_operator_single_qubit(self):
        """Test single-qubit Pauli operators."""
        # For n=1:
        # Index 0 → I
        P_I = get_pauli_operator(0, 1)
        assert P_I == Matrix([[1, 0], [0, 1]])

        # Index 1 → X
        P_X = get_pauli_operator(1, 1)
        assert P_X == msigma(1)

        # Index 2 → Y
        P_Y = get_pauli_operator(2, 1)
        assert P_Y == msigma(2)

        # Index 3 → Z
        P_Z = get_pauli_operator(3, 1)
        assert P_Z == msigma(3)

    def test_get_pauli_operator_two_qubits(self):
        """Test two-qubit operators."""
        # Index 1 → X ⊗ I
        P = get_pauli_operator(1, 2)
        assert P.shape == (4, 4)

        # Verify it's X ⊗ I by checking action on basis states
        # X ⊗ I should swap |00⟩ ↔ |10⟩ and |01⟩ ↔ |11⟩
        e00 = Matrix([1, 0, 0, 0])
        e10 = Matrix([0, 0, 1, 0])
        assert P * e00 == e10
        assert P * e10 == e00

    def test_get_pauli_operator_hermitian(self):
        """Test that Pauli operators are Hermitian."""
        for idx in range(16):  # Test all 2-qubit Paulis
            P = get_pauli_operator(idx, 2)
            assert P == P.H  # Hermitian: P = P†

    def test_get_pauli_operator_unitary(self):
        """Test that Pauli operators are unitary."""
        for idx in range(16):  # Test all 2-qubit Paulis
            P = get_pauli_operator(idx, 2)
            identity = Matrix.eye(4)
            product = P * P.H
            # Simplify element-wise and compare
            for i in range(4):
                for j in range(4):
                    assert product[i, j].simplify() == identity[i, j]


class TestNontrivialCount:
    """Test counting non-trivial qubits."""

    def test_count_nontrivial_qubits_identity(self):
        """Test that identity has 0 non-trivial qubits."""
        assert count_nontrivial_qubits(0, 9) == 0

    def test_count_nontrivial_qubits_single(self):
        """Test single-qubit operators."""
        # Index 1, 2, 3 → X_0, Y_0, Z_0 (one non-trivial qubit)
        assert count_nontrivial_qubits(1, 9) == 1
        assert count_nontrivial_qubits(2, 9) == 1
        assert count_nontrivial_qubits(3, 9) == 1

    def test_count_nontrivial_qubits_multiple(self):
        """Test multi-qubit operators."""
        # Index 5 = [1, 1, 0, ...] → X_0 X_1 (two non-trivial)
        assert count_nontrivial_qubits(5, 9) == 2

        # Index 9 = [1, 2, 0, ...] → X_0 Y_1 (two non-trivial)
        assert count_nontrivial_qubits(9, 9) == 2

        # Index 262143 = all 3's → all Z (nine non-trivial)
        assert count_nontrivial_qubits(262143, 9) == 9


class TestBasisPnt:
    """Test P_{n,t} basis extraction."""

    def test_get_basis_P_9_0(self):
        """Test P_{9,0}: only identity."""
        basis = get_basis_P_n_t(9, 0)
        assert basis == [0]
        assert len(basis) == 1

    def test_get_basis_P_9_1_size(self):
        """Test P_{9,1}: identity + 3*9 single-qubit Paulis."""
        basis = get_basis_P_n_t(9, 1)
        # Should have dimension 1 + 3*9 = 28
        assert len(basis) == 28

    def test_get_basis_P_9_1_contains_identity(self):
        """Test that P_{9,1} starts with identity."""
        basis = get_basis_P_n_t(9, 1)
        assert basis[0] == 0

    def test_get_basis_P_9_1_contains_single_qubit_paulis(self):
        """Test that P_{9,1} contains all single-qubit Paulis."""
        basis = get_basis_P_n_t(9, 1)

        # Check that indices 1, 2, 3 (X_0, Y_0, Z_0) are included
        assert 1 in basis
        assert 2 in basis
        assert 3 in basis

        # Check that all single-qubit operators are included
        for i in range(9):
            for pauli_idx in [1, 2, 3]:  # X, Y, Z
                idx = pauli_idx * (4 ** i)
                assert idx in basis

    def test_get_basis_P_9_1_excludes_two_qubit(self):
        """Test that P_{9,1} excludes two-qubit operators."""
        basis = get_basis_P_n_t(9, 1)

        # Index 5 = X_0 X_1 (two qubits) should not be in P_{9,1}
        assert 5 not in basis

    def test_get_basis_P_2_1(self):
        """Test P_{2,1} for smaller example."""
        basis = get_basis_P_n_t(2, 1)
        # Should have 1 + 3*2 = 7 elements
        assert len(basis) == 7

        # Should be: I, X_0, Y_0, Z_0, X_1, Y_1, Z_1
        # Indices: 0, 1, 2, 3, 4, 8, 12
        expected = [0, 1, 2, 3, 4, 8, 12]
        assert basis == expected
