"""
Isotropic extension algorithm for quantum error correction codes.

This module implements the algorithm from Grove Prop. 10.8 for extending
an isotropic code space M by one dimension while preserving isotropy.

Given:
- Code space M with isotropic vector u (e.g., |0_L⟩)
- Nondegenerate total error form D̂

The algorithm finds a hyperbolic partner v such that:
- D̂(u,u) = D̂(v,v) = 0 (both isotropic)
- D̂(u,v) = 1 (hyperbolic pair)

Then u+v is isotropic with respect to all error forms B̂_{λ,E,F},
and M' = M ⊕ span(u+v) is an isotropic extension.
"""

from sympy import Matrix, linsolve, symbols, simplify
from typing import Tuple, List, Optional

from .error_forms import compute_B_hat_lambda_EF
from .matrix_io import log


def find_hyperbolic_partner(u: Matrix, D_hat: Matrix) -> Tuple[Matrix, complex]:
    """
    Find a hyperbolic partner v for isotropic vector u.

    Given an isotropic vector u (D̂(u,u) = 0) and nondegenerate form D̂,
    constructs v such that (u,v) is a hyperbolic pair:
    - D̂(u,v) = 1
    - D̂(v,v) = 0

    Algorithm:
    1. Solve u†D̂w = 1 for w
    2. Compute b = -½⟨w|D̂|w⟩
    3. Set v = bu + w

    Args:
        u: Isotropic vector as column Matrix (dim × 1)
        D_hat: Total error form matrix (dim × dim)

    Returns:
        Tuple of (v, b) where:
        - v is the hyperbolic partner (dim × 1 Matrix)
        - b is the scalar coefficient used in construction

    Raises:
        ValueError: If no solution exists (D̂ is degenerate)
    """
    dim = D_hat.shape[0]

    # Create symbolic variables for w components
    w_vars = symbols(f'w0:{dim}')
    w_symbolic = Matrix(w_vars)

    # Set up equation: u†D̂w = 1
    # u†D̂w is a scalar (1×1 matrix), extract it
    lhs = (u.H * D_hat * w_symbolic)[0, 0]
    equation = lhs - 1

    # Solve for w
    solutions = linsolve([equation], w_vars)

    if not solutions:
        raise ValueError("No solution found: D̂ may be degenerate on span(u)")

    # Get the first solution (there may be infinitely many, we just need one)
    solution = list(solutions)[0]

    # Convert solution tuple to column vector
    w = Matrix([sol for sol in solution])

    # Check for free symbols (parameters) in the solution
    # These are parameters that weren't constrained by the equation
    free_symbols = w.free_symbols
    if free_symbols:
        # Substitute all free parameters with 0 to get a concrete solution
        substitutions = {sym: 0 for sym in free_symbols}
        w = w.subs(substitutions)

    # Compute D(w,w) = w†D̂w
    D_ww = (w.H * D_hat * w)[0, 0]

    # Compute b = -½ D(w,w)
    b = -simplify(D_ww) / 2

    # Construct v = bu + w
    v = b * u + w

    return v, b


def check_isotropy_on_error_form(
    vector: Matrix,
    E_idx: int,
    F_idx: int,
    n_qubits: int,
    ket_v: Matrix
) -> bool:
    """
    Check if a vector is isotropic with respect to B̂_{λ,E,F}.

    Computes B̂_{λ,E,F}(vector, vector) = vector†B̂_{λ,E,F}vector
    and checks if it equals zero.

    Args:
        vector: The vector to check (column Matrix)
        E_idx: Index for error operator E
        F_idx: Index for error operator F
        n_qubits: Number of qubits
        ket_v: Reference state for λ computation (e.g., |0_L⟩)

    Returns:
        True if vector is isotropic with respect to B̂_{λ,E,F}
    """
    # Compute B̂_{λ,E,F}
    B_hat = compute_B_hat_lambda_EF(E_idx, F_idx, n_qubits, ket_v)

    # Compute quadratic form: vector†B̂vector
    result = (vector.H * B_hat * vector)[0, 0]
    result_simplified = simplify(result)

    return result_simplified == 0


def verify_isotropy_on_all_error_forms(
    vector: Matrix,
    basis: List[int],
    n_qubits: int,
    ket_v: Matrix
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Verify that a vector is isotropic with respect to all B̂_{λ,E_i,E_j}.

    Checks the isotropy condition for all pairs (E_i, E_j) in the error basis.

    Args:
        vector: The vector to check (column Matrix)
        basis: List of error operator indices (from P_{n,t})
        n_qubits: Number of qubits
        ket_v: Reference state for λ computation (e.g., |0_L⟩)

    Returns:
        Tuple of (all_isotropic, failed_pairs) where:
        - all_isotropic is True if all checks pass
        - failed_pairs is list of (E_idx, F_idx) pairs that failed
    """
    failed_pairs = []

    for E_idx in basis:
        for F_idx in basis:
            is_isotropic = check_isotropy_on_error_form(
                vector, E_idx, F_idx, n_qubits, ket_v
            )

            if not is_isotropic:
                failed_pairs.append((E_idx, F_idx))

    all_isotropic = len(failed_pairs) == 0
    return all_isotropic, failed_pairs


def compute_isotropic_extension(
    u: Matrix,
    D_hat: Matrix,
    basis: List[int],
    n_qubits: int,
    ket_v: Matrix
) -> Optional[Matrix]:
    """
    Compute an isotropic extension of the code space M.

    Given an isotropic vector u ∈ M and the total error form D̂, finds a
    hyperbolic partner v and constructs the extension vector u+v.

    If u+v is isotropic with respect to all error forms B̂_{λ,E,F}, then
    M' = M ⊕ span(u+v) is an isotropic extension.

    Args:
        u: Isotropic vector from M (e.g., |0_L⟩)
        D_hat: Total error form matrix D̂
        basis: Error operator basis (from P_{n,t})
        n_qubits: Number of qubits
        ket_v: Reference state for λ computation (e.g., |0_L⟩)

    Returns:
        The extension vector u+v if successful, None otherwise
    """
    # Step 1: Find hyperbolic partner v
    v, b = find_hyperbolic_partner(u, D_hat)

    # Step 2: Construct extension vector
    extension_vector = u + v

    # Step 3: Verify isotropy with respect to all error forms
    all_isotropic, failed_pairs = verify_isotropy_on_all_error_forms(
        extension_vector, basis, n_qubits, ket_v
    )

    if all_isotropic:
        return extension_vector
    else:
        return None


def find_simultaneous_hyperbolic_partner(
    u: Matrix,
    basis: List[int],
    n_qubits: int,
    ket_v: Matrix,
    verbose: bool = False
) -> Matrix:
    """
    Find w such that B_{λ,E_i,E_j}(u,w) = 1 for all (E_i,E_j) pairs.

    This solves the stacked system:
        Q w = [1, 1, ..., 1]^T

    where Q is a (basis_size^2 × dim) matrix with rows q_{i,j} = u†B̂_{λ,E_i,E_j}.

    Args:
        u: Isotropic vector from M (e.g., |0_L⟩)
        basis: Error operator basis (from P_{n,t})
        n_qubits: Number of qubits
        ket_v: Reference state for λ computation (e.g., |0_L⟩)
        verbose: If True, print progress messages

    Returns:
        Vector w satisfying all constraints

    Raises:
        ValueError: If no solution exists
    """
    dim = 2 ** n_qubits
    basis_size = len(basis)
    num_constraints = basis_size * basis_size

    if verbose:
        log(f"Building stacked system Q w = 1...")
        log(f"  Dimension of w: {dim}")
        log(f"  Number of constraints: {num_constraints}")
        log(f"  System size: {num_constraints} × {dim}")
        log("")

    # Create symbolic variables for w components
    w_vars = symbols(f'w0:{dim}')
    w_symbolic = Matrix(w_vars)

    # Build the stacked system
    equations = []
    constraint_count = 0

    for i, E_idx in enumerate(basis):
        for j, F_idx in enumerate(basis):
            constraint_count += 1

            if verbose and constraint_count % 100 == 0:
                log(f"  Building constraint {constraint_count}/{num_constraints}...")

            # Compute B̂_{λ,E_i,E_j}
            B_hat = compute_B_hat_lambda_EF(E_idx, F_idx, n_qubits, ket_v)

            # Compute q_{i,j} = u†B̂_{λ,E_i,E_j}
            # This is a row vector (1 × dim)
            q_row = u.H * B_hat

            # Form equation: q_{i,j} · w = 1
            # q_row is 1×dim, w_symbolic is dim×1, result is scalar
            lhs = (q_row * w_symbolic)[0, 0]
            equation = lhs - 1

            equations.append(equation)

    if verbose:
        log("")
        log(f"All {num_constraints} constraints built!")
        log("")
        log("Solving stacked system...")

    # Solve the stacked system
    solutions = linsolve(equations, w_vars)

    if not solutions:
        raise ValueError("No solution found for stacked system")

    # Get the first solution
    solution = list(solutions)[0]

    # Convert solution tuple to column vector
    w = Matrix([sol for sol in solution])

    # Check for free symbols and substitute with 0
    free_symbols = w.free_symbols
    if free_symbols:
        if verbose:
            log(f"  Found {len(free_symbols)} free parameters, setting to 0...")
        substitutions = {sym: 0 for sym in free_symbols}
        w = w.subs(substitutions)

    if verbose:
        log("  Solution found!")
        log("")

    return w
