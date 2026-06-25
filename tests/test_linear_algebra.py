"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.linear_algebra: tridiagonal solver.
"""

import numpy as np
from numpy import array, diag
from numpy.linalg import solve
from numpy.testing import assert_allclose

from pynalgo.linear_algebra import (
    solver_tridiagonal,
    solver_pentadiagonal,
    solver_pentadiagonalF,
    solver_pentadiagonalFS,
)


def test_solver_tridiagonal_known():
    """Solve known 5x5 tridiagonal system, verify T*x = y."""
    _ = 5  # N
    a = array([1.0, 2.0, 1.0, 3.0])      # sub-diagonal (length N-1)
    b = array([4.0, 4.0, 4.0, 4.0, 4.0]) # main diagonal (length N)
    c = array([1.0, 2.0, 1.0, 3.0])      # super-diagonal (length N-1)
    y = array([1.0, 2.0, 3.0, 4.0, 5.0])

    x = solver_tridiagonal(a, b, c, y)

    # Reconstruct T and verify T*x = y
    T = (diag(b)
         + diag(a, k=-1)
         + diag(c, k=1))
    Tx = T @ x
    assert_allclose(Tx, y, atol=1e-14)


def test_solver_tridiagonal_vs_numpy():
    """Random well-conditioned tridiagonal,
    compare against numpy.linalg.solve."""
    rng = np.random.default_rng(42)
    N = 10
    # diagonally dominant for numerical stability
    a = rng.uniform(-0.5, 0.5, size=N - 1)
    b = rng.uniform(2.0, 5.0, size=N)
    c = rng.uniform(-0.5, 0.5, size=N - 1)
    y = rng.uniform(-1.0, 1.0, size=N)

    T = (diag(b) + diag(a, k=-1) + diag(c, k=1))

    x_pynalgo = solver_tridiagonal(a, b, c, y)
    x_numpy = solve(T, y)

    assert_allclose(x_pynalgo, x_numpy, atol=1e-12)


def test_solver_tridiagonal_simple_diagonal():
    """Identity-like tridiagonal: a=c=0, b=diag, x = y/b."""
    _ = 4  # N
    a = array([0.0, 0.0, 0.0])
    b = array([2.0, 3.0, 4.0, 5.0])
    c = array([0.0, 0.0, 0.0])
    y = array([1.0, 2.0, 3.0, 4.0])

    x = solver_tridiagonal(a, b, c, y)
    expected = y / b
    assert_allclose(x, expected, atol=1e-14)


def test_solver_pentadiagonal_known():
    """Solve known 8x8 pentadiagonal system, verify P*x = y.
    """
    # Main diagonal
    d = array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # 1st super-diagonal (a), length N-1
    a = array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # 2nd super-diagonal (b), length N-2
    b = array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # 1st sub-diagonal (c), length N-1
    c = array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # 2nd sub-diagonal (e), length N-2
    e = array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    x = solver_pentadiagonal(a, b, c, d, e, y)

    # Reconstruct P and verify P*x = y
    P = (diag(d)
         + diag(a, k=1)
         + diag(b, k=2)
         + diag(c, k=-1)
         + diag(e, k=-2))
    Px = P @ x
    assert_allclose(Px, y, atol=1e-14)


def test_solver_pentadiagonal_vs_numpy():
    """Random well-conditioned pentadiagonal,
    compare against numpy.linalg.solve.
    """
    rng = np.random.default_rng(42)
    N = 20
    # diagonally dominant for numerical stability
    a = rng.uniform(-0.3, 0.3, size=N - 1)
    b = rng.uniform(-0.2, 0.2, size=N - 2)
    c = rng.uniform(-0.3, 0.3, size=N - 1)
    d = rng.uniform(3.0, 7.0, size=N)
    e = rng.uniform(-0.2, 0.2, size=N - 2)
    y = rng.uniform(-1.0, 1.0, size=N)

    # Build dense matrix P from diagonals
    P = (diag(d) + diag(a, k=1) + diag(b, k=2)
         + diag(c, k=-1) + diag(e, k=-2))

    x_pynalgo = solver_pentadiagonal(a, b, c, d, e, y)
    x_numpy = solve(P, y)

    assert_allclose(x_pynalgo, x_numpy, atol=1e-12)


def test_solver_pentadiagonal_factored():
    """Verify factored solve matches direct solve
    for a random well-conditioned system.
    """
    rng = np.random.default_rng(99)
    N = 15
    a = rng.uniform(-0.3, 0.3, size=N - 1)
    b = rng.uniform(-0.2, 0.2, size=N - 2)
    c = rng.uniform(-0.3, 0.3, size=N - 1)
    d = rng.uniform(3.0, 7.0, size=N)
    e = rng.uniform(-0.2, 0.2, size=N - 2)
    y = rng.uniform(-1.0, 1.0, size=N)

    # Direct solve
    x_direct = solver_pentadiagonal(a, b, c, d, e, y)

    # Factored solve
    al, be, ga_mu, e_mu, i_mu = solver_pentadiagonalF(a, b, c, d, e)
    x_factored = solver_pentadiagonalFS(al, be, ga_mu, e_mu, i_mu, y)

    assert_allclose(x_factored, x_direct, atol=1e-14)
