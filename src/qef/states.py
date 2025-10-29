"""
Quantum state creation for error correction codes.
"""

from sympy import Matrix, sqrt
from sympy.physics.quantum import TensorProduct


def create_ghz_state() -> Matrix:
    """
    Create the 3-qubit GHZ-like state |000⟩ + |111⟩ (unnormalized).

    Returns:
        SymPy Matrix (8x1) representing |000⟩ + |111⟩
    """
    state = Matrix.zeros(8, 1)
    state[0] = 1  # |000⟩
    state[7] = 1  # |111⟩
    return state


def create_shor_logical_zero() -> Matrix:
    """
    Create the 9-qubit Shor code logical zero state.

    |0_L⟩ = (1/(2√2))(|000⟩ + |111⟩)^⊗3

    This is the tensor product of three copies of the GHZ-like state,
    normalized by 1/(2√2).

    Returns:
        SymPy Matrix (512x1) representing the normalized state |0_L⟩
    """
    # Create the 3-qubit GHZ state
    ghz = create_ghz_state()

    # Tensor product of three copies: ghz ⊗ ghz ⊗ ghz
    state = TensorProduct(ghz, ghz, ghz)

    # Normalize by 1/(2√2)
    normalization = 1 / (2 * sqrt(2))
    state = normalization * state

    return state


def create_shor_logical_one() -> Matrix:
    """
    Create the 9-qubit Shor code logical one state.

    |1_L⟩ = (1/(2√2))(|000⟩ - |111⟩)^⊗3

    Returns:
        SymPy Matrix (512x1) representing the normalized state |1_L⟩
    """
    # Create the 3-qubit state |000⟩ - |111⟩
    state_3 = Matrix.zeros(8, 1)
    state_3[0] = 1   # |000⟩
    state_3[7] = -1  # -|111⟩

    # Tensor product of three copies
    state = TensorProduct(state_3, state_3, state_3)

    # Normalize by 1/(2√2)
    normalization = 1 / (2 * sqrt(2))
    state = normalization * state

    return state
