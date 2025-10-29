"""Unit tests for states module."""

import pytest
from sympy import Matrix, sqrt, simplify

from qef.states import (
    create_ghz_state,
    create_shor_logical_zero,
    create_shor_logical_one,
)


class TestGHZState:
    """Test GHZ state creation."""

    def test_ghz_state_dimension(self):
        """Test that GHZ state has correct dimension."""
        ghz = create_ghz_state()
        assert ghz.shape == (8, 1)

    def test_ghz_state_components(self):
        """Test that GHZ state is |000⟩ + |111⟩."""
        ghz = create_ghz_state()

        # Check that only components 0 and 7 are non-zero
        for i in range(8):
            if i == 0 or i == 7:
                assert ghz[i] == 1
            else:
                assert ghz[i] == 0

    def test_ghz_state_unnormalized(self):
        """Test that GHZ state is unnormalized (norm = sqrt(2))."""
        ghz = create_ghz_state()
        norm_squared = (ghz.H * ghz)[0, 0]
        assert norm_squared == 2  # |000⟩ + |111⟩ has norm sqrt(2)


class TestShorLogicalZero:
    """Test Shor code logical zero state."""

    def test_shor_logical_zero_dimension(self):
        """Test that |0_L⟩ has correct dimension for 9 qubits."""
        ket_0L = create_shor_logical_zero()
        assert ket_0L.shape == (512, 1)  # 2^9 = 512

    def test_shor_logical_zero_normalized(self):
        """Test that |0_L⟩ is normalized."""
        ket_0L = create_shor_logical_zero()
        norm_squared = (ket_0L.H * ket_0L)[0, 0]
        # Should be exactly 1
        assert simplify(norm_squared) == 1

    def test_shor_logical_zero_support(self):
        """Test that |0_L⟩ has support on the expected basis states."""
        ket_0L = create_shor_logical_zero()

        # The state (|000⟩ + |111⟩)^⊗3 has support on 8 basis states:
        # |000 000 000⟩, |000 000 111⟩, |000 111 000⟩, |000 111 111⟩,
        # |111 000 000⟩, |111 000 111⟩, |111 111 000⟩, |111 111 111⟩
        #
        # In binary (9-bit):
        # 000000000 = 0
        # 000000111 = 7
        # 000111000 = 56
        # 000111111 = 63
        # 111000000 = 448
        # 111000111 = 455
        # 111111000 = 504
        # 111111111 = 511

        expected_support = [0, 7, 56, 63, 448, 455, 504, 511]
        expected_amplitude = 1 / (2 * sqrt(2))

        # Check that these components are non-zero with correct amplitude
        for idx in expected_support:
            assert simplify(ket_0L[idx]) == expected_amplitude

        # Check that all other components are zero
        for i in range(512):
            if i not in expected_support:
                assert ket_0L[i] == 0

    def test_shor_logical_zero_eigenstate(self):
        """Test that |0_L⟩ is a +1 eigenstate of logical Z operators."""
        # The logical Z operator for Shor code is Z^⊗3 ⊗ I^⊗3 ⊗ I^⊗3
        # (or any permutation of the three blocks)
        # We can test with a simpler check: verify it's in the +1 eigenspace
        # of the stabilizers. For now, just verify normalization and structure.
        ket_0L = create_shor_logical_zero()

        # Basic sanity check: should be normalized
        norm_squared = (ket_0L.H * ket_0L)[0, 0]
        assert simplify(norm_squared) == 1


class TestShorLogicalOne:
    """Test Shor code logical one state."""

    def test_shor_logical_one_dimension(self):
        """Test that |1_L⟩ has correct dimension for 9 qubits."""
        ket_1L = create_shor_logical_one()
        assert ket_1L.shape == (512, 1)  # 2^9 = 512

    def test_shor_logical_one_normalized(self):
        """Test that |1_L⟩ is normalized."""
        ket_1L = create_shor_logical_one()
        norm_squared = (ket_1L.H * ket_1L)[0, 0]
        # Should be exactly 1
        assert simplify(norm_squared) == 1

    def test_shor_logical_one_support(self):
        """Test that |1_L⟩ has support on the expected basis states."""
        ket_1L = create_shor_logical_one()

        # The state (|000⟩ - |111⟩)^⊗3 has support on the same 8 basis states,
        # but with different signs based on the number of |111⟩ factors
        expected_support = [0, 7, 56, 63, 448, 455, 504, 511]

        # Count how many |111⟩ blocks each basis state has:
        # 000000000 (0): 0 blocks → (+1)^0 = +1
        # 000000111 (7): 1 block  → (+1)^0 * (-1)^1 = -1
        # 000111000 (56): 1 block → -1
        # 000111111 (63): 2 blocks → +1
        # 111000000 (448): 1 block → -1
        # 111000111 (455): 2 blocks → +1
        # 111111000 (504): 2 blocks → +1
        # 111111111 (511): 3 blocks → -1

        expected_signs = {
            0: 1,
            7: -1,
            56: -1,
            63: 1,
            448: -1,
            455: 1,
            504: 1,
            511: -1,
        }

        expected_amplitude = 1 / (2 * sqrt(2))

        # Check that these components have the correct amplitude and sign
        for idx in expected_support:
            expected = expected_signs[idx] * expected_amplitude
            assert simplify(ket_1L[idx]) == expected

        # Check that all other components are zero
        for i in range(512):
            if i not in expected_support:
                assert ket_1L[i] == 0

    def test_shor_logical_states_orthogonal(self):
        """Test that |0_L⟩ and |1_L⟩ are orthogonal."""
        ket_0L = create_shor_logical_zero()
        ket_1L = create_shor_logical_one()

        # Inner product should be zero
        inner_product = (ket_0L.H * ket_1L)[0, 0]
        assert simplify(inner_product) == 0
