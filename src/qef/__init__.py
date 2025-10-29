"""
Quantum Error Forms

A library for computing Hermitian forms associated with quantum error correction codes.
"""

from .operators import (
    index_to_base4,
    base4_to_pauli_string,
    index_to_pauli_string,
    pauli_string_to_index,
    get_pauli_operator,
    count_nontrivial_qubits,
    get_basis_P_n_t,
)

from .states import (
    create_ghz_state,
    create_shor_logical_zero,
    create_shor_logical_one,
)

from .lambda_hat import (
    compute_lambda,
)

__version__ = "0.1.0"

__all__ = [
    "index_to_base4",
    "base4_to_pauli_string",
    "index_to_pauli_string",
    "pauli_string_to_index",
    "get_pauli_operator",
    "count_nontrivial_qubits",
    "get_basis_P_n_t",
    "create_ghz_state",
    "create_shor_logical_zero",
    "create_shor_logical_one",
    "compute_lambda",
]
