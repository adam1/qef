"""
Pauli operator utilities with base-4 indexing.

The base-4 indexing scheme maps integers to Pauli operators:
    0 → I (identity)
    1 → X
    2 → Y
    3 → Z

For n qubits, index s ∈ [0, 4^n - 1] maps to a tensor product of Paulis:
    s → [s]_4 (n-digit base-4 string) → W_0 ⊗ W_1 ⊗ ... ⊗ W_{n-1}

where digit position i (from right, 0-indexed) determines the Pauli at qubit i.

Example for n=9:
    1 → 000000001 → X ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I
    5 → 000000011 → X ⊗ X ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I ⊗ I
"""

from sympy import Matrix, I
from sympy.physics.quantum import TensorProduct
from sympy.physics.matrices import msigma
from typing import List, Tuple


def index_to_base4(index: int, n_qubits: int) -> List[int]:
    """
    Convert an integer index to its base-4 representation with n_qubits digits.

    Args:
        index: Integer in range [0, 4^n_qubits - 1]
        n_qubits: Number of qubits (number of base-4 digits)

    Returns:
        List of n_qubits integers in {0, 1, 2, 3}, where position i (from right)
        corresponds to qubit i.

    Example:
        >>> index_to_base4(1, 9)
        [1, 0, 0, 0, 0, 0, 0, 0, 0]  # rightmost digit is qubit 0
        >>> index_to_base4(5, 9)
        [1, 1, 0, 0, 0, 0, 0, 0, 0]  # 5 = 1*4^0 + 1*4^1
    """
    if index < 0 or index >= 4**n_qubits:
        raise ValueError(f"Index {index} out of range for {n_qubits} qubits")

    digits = []
    for _ in range(n_qubits):
        digits.append(index % 4)
        index //= 4

    return digits


def base4_to_pauli_string(base4_digits: List[int]) -> str:
    """
    Convert base-4 digits to a Pauli string.

    Args:
        base4_digits: List of integers in {0, 1, 2, 3}

    Returns:
        String of Pauli operators, e.g., "XII III III"

    Example:
        >>> base4_to_pauli_string([1, 0, 0, 0, 0, 0, 0, 0, 0])
        'XII III III'
    """
    pauli_chars = ['I', 'X', 'Y', 'Z']

    # Convert each digit to Pauli character
    pauli_list = [pauli_chars[d] for d in base4_digits]

    # Group into sets of 3 for readability (for 9-qubit case)
    n = len(pauli_list)
    if n == 9:
        return f"{pauli_list[0]}{pauli_list[1]}{pauli_list[2]} " + \
               f"{pauli_list[3]}{pauli_list[4]}{pauli_list[5]} " + \
               f"{pauli_list[6]}{pauli_list[7]}{pauli_list[8]}"
    else:
        return ''.join(pauli_list)


def index_to_pauli_string(index: int, n_qubits: int) -> str:
    """
    Convert an index directly to a Pauli string.

    Args:
        index: Integer in range [0, 4^n_qubits - 1]
        n_qubits: Number of qubits

    Returns:
        String representation of the Pauli operator

    Example:
        >>> index_to_pauli_string(1, 9)
        'XII III III'
    """
    base4 = index_to_base4(index, n_qubits)
    return base4_to_pauli_string(base4)


def pauli_string_to_index(pauli_string: str, n_qubits: int) -> int:
    """
    Convert a Pauli string back to its index.

    Args:
        pauli_string: String of Pauli operators (spaces ignored)
        n_qubits: Number of qubits

    Returns:
        Integer index

    Example:
        >>> pauli_string_to_index('XII III III', 9)
        1
    """
    # Remove spaces
    pauli_string = pauli_string.replace(' ', '')

    if len(pauli_string) != n_qubits:
        raise ValueError(f"Pauli string length {len(pauli_string)} != n_qubits {n_qubits}")

    pauli_to_digit = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

    index = 0
    for i, char in enumerate(pauli_string):
        if char not in pauli_to_digit:
            raise ValueError(f"Invalid Pauli character: {char}")
        digit = pauli_to_digit[char]
        index += digit * (4 ** i)

    return index


def get_pauli_operator(index: int, n_qubits: int) -> Matrix:
    """
    Get the Pauli operator matrix corresponding to an index.

    Args:
        index: Integer in range [0, 4^n_qubits - 1]
        n_qubits: Number of qubits

    Returns:
        SymPy Matrix of dimension 2^n_qubits × 2^n_qubits

    Example:
        >>> P = get_pauli_operator(1, 2)  # X ⊗ I
        >>> P.shape
        (4, 4)
    """
    # Define single-qubit Pauli matrices
    pauli_I = Matrix([[1, 0], [0, 1]])
    pauli_X = msigma(1)
    pauli_Y = msigma(2)
    pauli_Z = msigma(3)

    pauli_matrices = [pauli_I, pauli_X, pauli_Y, pauli_Z]

    # Get base-4 representation
    base4 = index_to_base4(index, n_qubits)

    # Build tensor product: W_0 ⊗ W_1 ⊗ ... ⊗ W_{n-1}
    result = pauli_matrices[base4[0]]
    for i in range(1, n_qubits):
        result = TensorProduct(result, pauli_matrices[base4[i]])

    return result


def count_nontrivial_qubits(index: int, n_qubits: int) -> int:
    """
    Count the number of qubits affected non-trivially (not identity).

    Args:
        index: Integer in range [0, 4^n_qubits - 1]
        n_qubits: Number of qubits

    Returns:
        Number of non-identity Paulis

    Example:
        >>> count_nontrivial_qubits(0, 9)  # I^9
        0
        >>> count_nontrivial_qubits(1, 9)  # X_0
        1
        >>> count_nontrivial_qubits(5, 9)  # X_0 X_1
        2
    """
    base4 = index_to_base4(index, n_qubits)
    return sum(1 for d in base4 if d != 0)


def get_basis_P_n_t(n_qubits: int, t: int) -> List[int]:
    """
    Get the ordered basis P_{n,t}: all Pauli operators affecting at most t qubits.

    This is a subset of the full Pauli basis P_n, with ordering inherited from
    the base-4 indexing.

    Args:
        n_qubits: Number of qubits
        t: Maximum number of qubits affected non-trivially

    Returns:
        List of indices in the full basis that belong to P_{n,t}

    Example:
        >>> get_basis_P_n_t(9, 1)[:5]
        [0, 1, 2, 3, 4]  # I^9, X_0, Y_0, Z_0, X_1
        >>> len(get_basis_P_n_t(9, 1))
        28  # 1 + 3*9
    """
    basis = []
    for index in range(4**n_qubits):
        if count_nontrivial_qubits(index, n_qubits) <= t:
            basis.append(index)

    return basis
