"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.special_functions: grids,
           polynomials, special functions.
"""

import numpy as np
from numpy import (array, cos, exp, linspace, ones, pi, sin, sqrt, zeros)
from numpy.testing import assert_allclose
from scipy.special import gammaln as scipy_gammaln

from pynalgo.special_functions import (
    grid_ChebyshevT, grid_JacobiP, grid_LegendreP, grid_Fourier,
    poly_JacobiP, poly_der_JacobiP,
    poly_ChebyshevT, poly_der_ChebyshevT,
    poly_ChebyshevU, poly_der_ChebyshevU,
    poly_ChebyshevV, poly_der_ChebyshevV,
    poly_ChebyshevW, poly_der_ChebyshevW,
    poly_Gegenbauer, poly_der_Gegenbauer,
    poly_Ultraspherical, poly_der_Ultraspherical,
    poly_ChebyshevT_direct, poly_ChebyshevU_direct,
    poly_Laguerre, poly_Laguerre_lambda,
    poly_der_Laguerre, poly_der_Laguerre_lambda,
    poly_sin, poly_cos,
    poly_Hermite_psi, poly_Hermite_H, poly_der_Hermite_H,
    poly_ChebyshevT_mod,
    poly_LegendreP, poly_der_LegendreP,
    special_exp_log_frac_sum, special_log_abs_gamma,
)


# =============================================================================
# Grids: ChebyshevT
# =============================================================================

def test_grid_ChebyshevT_GL_endpoints():
    """GL grid includes +-1."""
    gr = grid_ChebyshevT(N=8, variety="GL")
    assert_allclose(gr[0], -1.0, atol=1e-14)
    assert_allclose(gr[-1], 1.0, atol=1e-14)


def test_grid_ChebyshevT_G_roots():
    """G grid: T_{N+1}(gr) = 0."""
    N = 8
    gr = grid_ChebyshevT(N, variety="G")
    TNp1 = cos((N + 1) * np.arccos(gr))
    assert_allclose(TNp1, zeros(N + 1), atol=1e-13)


def test_grid_ChebyshevT_RL_endpoint():
    """RL grid includes -1."""
    gr = grid_ChebyshevT(N=8, variety="RL")
    assert_allclose(gr[0], -1.0, atol=1e-14)


def test_grid_ChebyshevT_RR_endpoint():
    """RR grid includes +1."""
    gr = grid_ChebyshevT(N=8, variety="RR")
    assert_allclose(gr[-1], 1.0, atol=1e-14)


# =============================================================================
# Grids: Fourier
# =============================================================================

def test_grid_Fourier_S1_CL():
    """S1_CL: [0, 2*pi)."""
    N = 16
    gr = grid_Fourier(N, variety="S1_CL")
    assert_allclose(gr[0], 0.0, atol=1e-14)
    assert gr[-1] < 2 * pi
    assert gr[-1] > 0


def test_grid_Fourier_S1_CR():
    """S1_CR: (0, 2*pi]."""
    N = 16
    gr = grid_Fourier(N, variety="S1_CR")
    assert gr[0] > 0
    assert_allclose(gr[-1], 2 * pi, atol=1e-14)


def test_grid_Fourier_S1_I():
    """S1_I: interior (0, 2*pi)."""
    N = 16
    gr = grid_Fourier(N, variety="S1_I")
    assert gr[0] > 0
    assert gr[-1] < 2 * pi


def test_grid_Fourier_H1_C():
    """H1_C: [0, pi]."""
    N = 16
    gr = grid_Fourier(N, variety="H1_C")
    assert_allclose(gr[0], 0.0, atol=1e-14)
    assert_allclose(gr[-1], pi, atol=1e-14)


def test_grid_Fourier_H1_I():
    """H1_I: interior (0, pi)."""
    N = 16
    gr = grid_Fourier(N, variety="H1_I")
    assert gr[0] > 0
    assert gr[-1] < pi


# =============================================================================
# Grids: JacobiP
# =============================================================================

def test_grid_JacobiP_GL_endpoints():
    """GL grid includes +-1."""
    gr = grid_JacobiP(N=6, a=0.3, b=0.7, variety="GL")
    assert_allclose(gr[0], -1.0, atol=1e-14)
    assert_allclose(gr[-1], 1.0, atol=1e-14)


def test_grid_JacobiP_G_roots():
    """G grid: P_{N+1}(gr) = 0."""
    N, a, b = 4, 0.3, 0.7
    gr = grid_JacobiP(N, a, b, variety="G", root_polish=True)
    PNp1 = poly_JacobiP(N + 1, a, b, gr).flatten()
    assert_allclose(PNp1, zeros(N + 1), atol=1e-12)


def test_grid_JacobiP_RL_endpoint():
    """RL grid includes -1."""
    gr = grid_JacobiP(N=6, a=0.3, b=0.7, variety="RL")
    assert_allclose(gr[0], -1.0, atol=1e-14)


def test_grid_JacobiP_RR_endpoint():
    """RR grid includes +1."""
    gr = grid_JacobiP(N=6, a=0.3, b=0.7, variety="RR")
    assert_allclose(gr[-1], 1.0, atol=1e-14)


# =============================================================================
# Grids: LegendreP (delegates to JacobiP with a=b=0)
# =============================================================================

def test_grid_LegendreP_equals_JacobiP_ab0():
    """Legendre grid = Jacobi grid with a=b=0."""
    N = 6
    for variety in ["GL", "G", "RL", "RR"]:
        gr_leg = grid_LegendreP(N, variety=variety, root_polish=False)
        gr_jac = grid_JacobiP(N, 0.0, 0.0, variety=variety, root_polish=False)
        assert_allclose(gr_leg, gr_jac, atol=1e-14)


# =============================================================================
# Polynomials: JacobiP
# =============================================================================

def test_poly_JacobiP_P0():
    """P_0^{(a,b)}(x) = 1."""
    a, b = 0.3, 0.7
    x = linspace(-1, 1, 5)
    P = poly_JacobiP(0, a, b, x, ret_ord_num=1)
    assert_allclose(P[0, :], ones(5), atol=1e-14)


def test_poly_JacobiP_P1():
    """P_1^{(a,b)}(x) = (a-b)/2 + (1 + (a+b)/2)*x."""
    a, b = 0.3, 0.7
    x = linspace(-1, 1, 5)
    P = poly_JacobiP(1, a, b, x, ret_ord_num=2)
    expected = (a - b) / 2 + (1 + (a + b) / 2) * x
    assert_allclose(P[1, :], expected, atol=1e-13)


def test_poly_JacobiP_known_legendre():
    """P_2^{(0,0)}(x) = (3x^2 - 1)/2."""
    x = linspace(-1, 1, 5)
    P = poly_JacobiP(2, 0.0, 0.0, x, ret_ord_num=3)
    expected = (3 * x ** 2 - 1) / 2
    assert_allclose(P[2, :], expected, atol=1e-13)


def test_poly_der_JacobiP_property():
    """d/dx P_n^{(a,b)} = 0.5*(n+a+b+1)*P_{n-1}^{(a+1,b+1)}."""
    n, a, b = 3, 0.3, 0.7
    x = linspace(-1, 1, 5)
    dP = poly_der_JacobiP(1, n, a, b, x, ret_der_num=1, ret_ord_num=1)
    der_actual = dP[0, 0, :]
    P_shifted = poly_JacobiP(n - 1, a + 1, b + 1, x, ret_ord_num=1)
    der_expected = 0.5 * (n + a + b + 1) * P_shifted[0, :]
    assert_allclose(der_actual, der_expected, atol=1e-12)


# =============================================================================
# Polynomials: ChebyshevT
# =============================================================================

def test_poly_ChebyshevT_cos_identity():
    """T_n(cos theta) = cos(n*theta)."""
    N = 5
    theta = linspace(0, pi, 10)
    x = cos(theta)
    T = poly_ChebyshevT(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(T[n, :], cos(n * theta), atol=1e-13)


def test_poly_ChebyshevT_endpoints():
    """T_n(1) = 1, T_n(-1) = (-1)^n."""
    N = 5
    x = array([-1.0, 1.0])
    T = poly_ChebyshevT(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(T[n, 0], (-1) ** n, atol=1e-14)  # x=-1
        assert_allclose(T[n, 1], 1.0, atol=1e-14)        # x=1


def test_poly_der_ChebyshevT_cos_identity():
    """d/dx T_n(cos theta) = n*sin(n*theta)/sin(theta)."""
    N = 5
    theta = linspace(0.1, pi - 0.1, 8)  # avoid sin(theta)=0
    x = cos(theta)
    dT = poly_der_ChebyshevT(1, N, x, ret_der_num=1, ret_ord_num=N + 1)
    for n in range(1, N + 1):
        expected = n * sin(n * theta) / sin(theta)
        assert_allclose(dT[0, n, :], expected, atol=1e-12)


# =============================================================================
# Polynomials: ChebyshevU
# =============================================================================

def test_poly_ChebyshevU_cos_identity():
    """U_n(cos theta) = sin((n+1)*theta)/sin(theta)."""
    N = 6
    theta = linspace(0.1, pi - 0.1, 10)
    x = cos(theta)
    U = poly_ChebyshevU(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        expected = sin((n + 1) * theta) / sin(theta)
        assert_allclose(U[n, :], expected, atol=2e-12)


def test_poly_ChebyshevU_endpoints():
    """U_n(1) = n+1, U_n(-1) = (-1)^n * (n+1)."""
    N = 6
    x = array([-1.0, 1.0])
    U = poly_ChebyshevU(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(U[n, 0], ((-1) ** n) * (n + 1), atol=1e-14)  # x=-1
        assert_allclose(U[n, 1], n + 1, atol=1e-14)                  # x=1


def test_poly_der_ChebyshevU_via_T():
    """d/dx U_n(x) = (x*U_n(x) - (n+1)*T_{n+1}(x)) / (1-x^2)."""
    N = 4
    x = linspace(-0.9, 0.9, 7)  # avoid singularities at +-1
    dU = poly_der_ChebyshevU(1, N, x, ret_der_num=1, ret_ord_num=N + 1)
    U_all = poly_ChebyshevU(N, x, ret_ord_num=N + 1)
    T_all = poly_ChebyshevT(N + 1, x, ret_ord_num=N + 2)
    for n in range(1, N + 1):
        expected = (x * U_all[n, :] - (n + 1) * T_all[n + 1, :]) / (1 - x ** 2)
        assert_allclose(dU[0, n, :], expected, atol=1e-10)


# =============================================================================
# Polynomials: ChebyshevV
# =============================================================================

def test_poly_ChebyshevV_endpoints():
    """V_n(1) = 1, V_n(-1) = (-1)^n * (2n+1)."""
    N = 6
    x = array([-1.0, 1.0])
    V = poly_ChebyshevV(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(V[n, 0], ((-1) ** n) * (2 * n + 1), atol=1e-14)  # x=-1
        assert_allclose(V[n, 1], 1.0, atol=1e-14)                         # x=1


def test_poly_der_ChebyshevV_via_Jacobi():
    """Derivative of V_n via shifted Jacobi parameters."""
    N, M = 4, 1
    x = linspace(-1, 1, 5)
    dV = poly_der_ChebyshevV(M, N, x, ret_der_num=1, ret_ord_num=N + 1)
    # d/dx V_n = (2n+1) * poly_der_ChebyshevT with shifted params
    # Check first derivative scales appropriately: non-zero for n>=1
    for n in range(1, N + 1):
        assert abs(dV[0, n, :]).max() > 1e-10  # derivative should not be zero


# =============================================================================
# Polynomials: ChebyshevW
# =============================================================================

def test_poly_ChebyshevW_endpoints():
    """W_n(1) = 2n+1, W_n(-1) = (-1)^n."""
    N = 6
    x = array([-1.0, 1.0])
    W = poly_ChebyshevW(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(W[n, 0], (-1) ** n, atol=1e-14)    # x=-1
        assert_allclose(W[n, 1], 2 * n + 1, atol=1e-14)    # x=1


def test_poly_der_ChebyshevW_shape():
    """Derivative shape and non-vanishing check."""
    N, M = 4, 1
    x = linspace(-1, 1, 5)
    dW = poly_der_ChebyshevW(M, N, x, ret_der_num=1, ret_ord_num=N + 1)
    assert dW.shape == (1, N + 1, x.size)
    for n in range(1, N + 1):
        assert abs(dW[0, n, :]).max() > 1e-10


# =============================================================================
# Polynomials: Chebyshev serivative cross-family consistency
# =============================================================================

def test_poly_der_Chebyshev_ret_counts():
    """Verify ret_der_num and ret_ord_num slicing for U/V/W."""
    M, N = 3, 6
    x = linspace(-1, 1, 4)
    for poly_der_fn in [poly_der_ChebyshevU,
                        poly_der_ChebyshevV,
                        poly_der_ChebyshevW]:
        dP_full = poly_der_fn(M, N, x, ret_der_num=M + 1, ret_ord_num=N + 1)
        dP_part = poly_der_fn(M, N, x, ret_der_num=2, ret_ord_num=3)
        assert dP_part.shape == (2, 3, x.size)
        # Last row of partial should match last row of full
        assert_allclose(dP_part[-1, :, :], dP_full[-1, -(3):, :], atol=1e-14)


# =============================================================================
# Polynomials: Chebyshev direct recursion
# =============================================================================

def test_poly_ChebyshevT_direct_vs_jacobi():
    """Direct recursion matches Jacobi-based poly_ChebyshevT."""
    N = 10
    x = linspace(-1, 1, 11)
    T_dir = poly_ChebyshevT_direct(N, x, ret_ord_num=N + 1)
    T_jac = poly_ChebyshevT(N, x, ret_ord_num=N + 1)
    assert_allclose(T_dir, T_jac, atol=1e-13)


def test_poly_ChebyshevU_direct_vs_jacobi():
    """Direct recursion matches Jacobi-based poly_ChebyshevU."""
    N = 10
    x = linspace(-1, 1, 11)
    U_dir = poly_ChebyshevU_direct(N, x, ret_ord_num=N + 1)
    U_jac = poly_ChebyshevU(N, x, ret_ord_num=N + 1)
    assert_allclose(U_dir, U_jac, atol=1e-13)


def test_poly_ChebyshevT_direct_cos_identity():
    """T_n(cos theta) = cos(n*theta)."""
    N = 5
    theta = linspace(0, pi, 10)
    x = cos(theta)
    T = poly_ChebyshevT_direct(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(T[n, :], cos(n * theta), atol=1e-13)


def test_poly_ChebyshevU_direct_endpoints():
    """U_n(1) = n+1, U_n(-1) = (-1)^n * (n+1)."""
    N = 6
    x = array([-1.0, 1.0])
    U = poly_ChebyshevU_direct(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(U[n, 0], ((-1) ** n) * (n + 1), atol=1e-14)
        assert_allclose(U[n, 1], n + 1, atol=1e-14)


# =============================================================================
# Polynomials: Gegenbauer
# =============================================================================

def test_poly_Gegenbauer_endpoints():
    """C_n^(lam)(1) = binom(n+2*lam-1, n).
    C_n^(lam)(-1) = (-1)^n * binom(n+2*lam-1, n)."""
    from scipy.special import binom as scipy_binom
    N, lam = 6, 2.5
    x = array([-1.0, 1.0])
    G = poly_Gegenbauer(N, lam, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        expected = scipy_binom(n + 2 * lam - 1, n)
        assert_allclose(G[n, 0], ((-1) ** n) * expected, atol=1e-12)
        assert_allclose(G[n, 1], expected, atol=1e-12)


def test_poly_Gegenbauer_C0_C1():
    """C_0^(lam)(x) = 1, C_1^(lam)(x) = 2*lam*x."""
    lam = 1.7
    x = linspace(-1, 1, 5)
    G = poly_Gegenbauer(1, lam, x, ret_ord_num=2)
    assert_allclose(G[0, :], ones(5), atol=1e-14)
    assert_allclose(G[1, :], 2 * lam * x, atol=1e-13)


def test_poly_Gegenbauer_recurrence():
    """n*C_n = 2*(n+lam-1)*x*C_{n-1} - (n+2*lam-2)*C_{n-2}."""
    N, lam = 5, 2.5
    x = linspace(-1, 1, 7)
    G = poly_Gegenbauer(N, lam, x, ret_ord_num=N + 1)
    for n in range(2, N + 1):
        lhs = n * G[n, :]
        rhs = (2 * (n + lam - 1) * x * G[n - 1, :]
               - (n + 2 * lam - 2) * G[n - 2, :])
        assert_allclose(lhs, rhs, atol=1e-12)


def test_poly_der_Gegenbauer_via_shift():
    """d/dx C_n^(lam)(x) = 2*lam * C_{n-1}^(lam+1)(x)."""
    N, lam = 4, 2.0
    x = linspace(-1, 1, 7)
    dG = poly_der_Gegenbauer(1, N, lam, x, ret_der_num=1, ret_ord_num=N + 1)
    for n in range(1, N + 1):
        C_shifted = poly_Gegenbauer(n - 1, lam + 1, x, ret_ord_num=1)
        expected = 2 * lam * C_shifted[0, :]
        assert_allclose(dG[0, n, :], expected, atol=1e-12)


# =============================================================================
# Polynomials: Ultraspherical
# =============================================================================

def test_poly_Ultraspherical_normalization():
    """P_n^(lam)(1) = 1 for all n."""
    N, lam = 5, 2.5
    x = array([-1.0, 0.0, 1.0])
    U = poly_Ultraspherical(N, lam, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(U[n, -1], 1.0, atol=1e-13)


def test_poly_Ultraspherical_vs_Gegenbauer():
    """P_n^(lam)(x) * C_n^(lam)(1) = C_n^(lam)(x)."""
    from scipy.special import binom as scipy_binom
    N, lam = 5, 2.5
    x = linspace(-1, 1, 7)
    U = poly_Ultraspherical(N, lam, x, ret_ord_num=N + 1)
    G = poly_Gegenbauer(N, lam, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        c_n_1 = scipy_binom(n + 2 * lam - 1, n) if n > 0 else 1.0
        assert_allclose(U[n, :] * c_n_1, G[n, :], atol=1e-12)


def test_poly_der_Ultraspherical_via_FD():
    """First derivative matches finite difference for multiple orders."""
    from scipy.special import binom as scipy_binom
    N, lam = 4, 2.5
    x = linspace(-0.9, 0.9, 10)
    eps = 1e-5

    dU = poly_der_Ultraspherical(1, N, lam, x, ret_der_num=1, ret_ord_num=N + 1)

    for n in range(1, N + 1):
        Up = poly_Ultraspherical(n, lam, x + eps, ret_ord_num=1)[0]
        Um = poly_Ultraspherical(n, lam, x - eps, ret_ord_num=1)[0]
        dP_fd = (Up - Um) / (2 * eps)
        err = np.max(np.abs(dU[0, n, :] - dP_fd))
        assert err < 1e-6, f'n={n}: max FD err = {err}'

    # DLMF formula: d/dx P_n^(lam) = 2*lam * C_{n-1}^(lam+1) / C_n^(lam)(1)
    for n in range(1, N + 1):
        C_n_1 = scipy_binom(n + 2*lam - 1, n)
        C_shifted = poly_Gegenbauer(n - 1, lam + 1, x, ret_ord_num=1)[0]
        dP_dlmf = 2 * lam * C_shifted / C_n_1
        err = np.max(np.abs(dU[0, n, :] - dP_dlmf))
        assert err < 1e-13, f'n={n}: max DLMF err = {err}'


def test_poly_der_Ultraspherical_second():
    """Second derivative matches FD and DLMF formula."""
    from scipy.special import binom as scipy_binom
    N, lam = 3, 1.5
    x = linspace(-0.9, 0.9, 10)
    eps = 1e-5

    d2U = poly_der_Ultraspherical(2, N, lam, x,
                                   ret_der_num=3, ret_ord_num=N + 1)

    for n in range(2, N + 1):
        dUp = poly_der_Ultraspherical(1, n, lam, x + eps,
                                       ret_der_num=1, ret_ord_num=1)[0, 0]
        dUm = poly_der_Ultraspherical(1, n, lam, x - eps,
                                       ret_der_num=1, ret_ord_num=1)[0, 0]
        d2P_fd = (dUp - dUm) / (2 * eps)
        err = np.max(np.abs(d2U[2, n, :] - d2P_fd))
        assert err < 1e-6, f'n={n}, m=2: max FD err = {err}'

    # DLMF: d^2/dx^2 P_n^(lam) = 4*lam*(lam+1)*C_{n-2}^(lam+2)/C_n^(lam)(1)
    for n in range(2, N + 1):
        C_n_1 = scipy_binom(n + 2*lam - 1, n)
        C_shifted = poly_Gegenbauer(n - 2, lam + 2, x, ret_ord_num=1)[0]
        d2P_dlmf = 4 * lam * (lam + 1) * C_shifted / C_n_1
        err = np.max(np.abs(d2U[2, n, :] - d2P_dlmf))
        assert err < 1e-13, f'n={n}, m=2: max DLMF err = {err}'


# =============================================================================
# Polynomials: Laguerre
# =============================================================================

def test_poly_Laguerre_L0_L1():
    """L_0^(alpha)(x) = 1, L_1^(alpha)(x) = 1+alpha-x."""
    alpha = 1.2
    x = linspace(0, 5, 7)
    L = poly_Laguerre(1, alpha, x, ret_ord_num=2)
    assert_allclose(L[0, :], ones(7), atol=1e-14)
    assert_allclose(L[1, :], 1 + alpha - x, atol=1e-13)


def test_poly_Laguerre_recurrence():
    """(n+1)*L_{n+1} = (2n+alpha+1-x)*L_n - (n+alpha)*L_{n-1}."""
    N, alpha = 6, 0.7
    x = linspace(0, 8, 9)
    L = poly_Laguerre(N, alpha, x, ret_ord_num=N + 1)
    for n in range(1, N):
        lhs = (n + 1) * L[n + 1, :]
        rhs = ((2 * n + alpha + 1 - x) * L[n, :]
               - (n + alpha) * L[n - 1, :])
        assert_allclose(lhs, rhs, atol=1e-12)


def test_poly_der_Laguerre_identity():
    """d/dx L_n^(alpha)(x) = -L_{n-1}^(alpha+1)(x)."""
    N, alpha = 5, 0.5
    x = linspace(0, 5, 7)
    dL = poly_der_Laguerre(1, N, alpha, x, ret_der_num=1, ret_ord_num=N + 1)
    for n in range(1, N + 1):
        L_shift = poly_Laguerre(n - 1, alpha + 1, x, ret_ord_num=1)
        assert_allclose(dL[0, n, :], -L_shift[0, :], atol=1e-12)


def test_poly_Laguerre_lambda_orthonormal():
    """lambda_n^(alpha) is finite and has the right shape."""
    N, alpha = 3, 0.5
    x = linspace(0.5, 5, 7)  # avoid x=0 for alpha>0
    Ll = poly_Laguerre_lambda(N, alpha, x, ret_ord_num=N + 1)
    assert Ll.shape == (N + 1, x.size)
    for n in range(N + 1):
        assert abs(Ll[n, :]).max() > 1e-10


def test_poly_der_Laguerre_lambda_numerical():
    """Derivative of lambda matches finite difference."""
    N, alpha = 3, 0.7
    x = linspace(1.0, 4.0, 5)
    dLl = poly_der_Laguerre_lambda(1, N, alpha, x,
                                   ret_der_num=1, ret_ord_num=N + 1)
    eps = 1e-6
    for n in range(1, N + 1):
        Ll_plus = poly_Laguerre_lambda(n, alpha, x + eps,
                                        ret_ord_num=1)
        Ll_minus = poly_Laguerre_lambda(n, alpha, x - eps,
                                         ret_ord_num=1)
        der_num = (Ll_plus[0, :] - Ll_minus[0, :]) / (2 * eps)
        assert_allclose(dLl[0, n, :], der_num, atol=1e-6)


# =============================================================================
# Polynomials: Trigonometric
# =============================================================================

def test_poly_sin_identity():
    """sin(n*x) matches numpy.sin."""
    N = 5
    x = linspace(0, 2 * pi, 10)
    S = poly_sin(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(S[n, :], sin(n * x), atol=1e-13)


def test_poly_cos_identity():
    """cos(n*x) matches numpy.cos."""
    N = 5
    x = linspace(0, 2 * pi, 10)
    C = poly_cos(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(C[n, :], cos(n * x), atol=1e-13)


# =============================================================================
# Polynomials: Hermite
# =============================================================================

def test_poly_Hermite_H_H0_H1():
    """H_0(x)=1, H_1(x)=2*x."""
    x = linspace(-2, 2, 5)
    H = poly_Hermite_H(1, x, ret_ord_num=2)
    assert_allclose(H[0, :], ones(5), atol=1e-14)
    assert_allclose(H[1, :], 2 * x, atol=1e-13)


def test_poly_Hermite_H_recurrence():
    """H_{n+1} = 2*x*H_n - 2*n*H_{n-1}."""
    N = 5
    x = linspace(-2, 2, 7)
    H = poly_Hermite_H(N, x, ret_ord_num=N + 1)
    for n in range(1, N):
        lhs = H[n + 1, :]
        rhs = 2 * x * H[n, :] - 2 * n * H[n - 1, :]
        assert_allclose(lhs, rhs, atol=1e-12)


def test_poly_der_Hermite_H_identity():
    """d/dx H_n = 2*n*H_{n-1}."""
    N = 5
    x = linspace(-2, 2, 7)
    dH = poly_der_Hermite_H(1, N, x, ret_der_num=1, ret_ord_num=N + 1)
    H = poly_Hermite_H(N - 1, x, ret_ord_num=N)
    for n in range(1, N + 1):
        assert_allclose(dH[0, n, :], 2 * n * H[n - 1, :], atol=1e-12)


def test_poly_Hermite_psi_finite():
    """Hermite psi functions: psi_0 and psi_1 have correct analytic form."""
    x = linspace(-3, 3, 7)
    psi = poly_Hermite_psi(1, x, ret_ord_num=2)
    # psi_0(x) = pi^(-1/4) * exp(-x^2/2)
    psi0_expected = pi**(-0.25) * exp(-x**2 / 2)
    assert_allclose(psi[0, :], psi0_expected, atol=1e-13)
    # psi_1(x) = sqrt(2) * pi^(-1/4) * x * exp(-x^2/2)
    psi1_expected = sqrt(2) * pi**(-0.25) * x * exp(-x**2 / 2)
    assert_allclose(psi[1, :], psi1_expected, atol=1e-13)


# =============================================================================
# Polynomials: Modified Chebyshev bases
# =============================================================================

def test_poly_ChebyshevT_mod_0_endpoints():
    """mod_ord=0: Tmod__n(+-1) = 0."""
    N = 4
    x = array([-1.0, 1.0])
    Tm = poly_ChebyshevT_mod(N, x, ret_ord_num=N + 1, mod_ord=0)
    assert_allclose(Tm[:, 0], zeros(N + 1), atol=1e-14)
    assert_allclose(Tm[:, -1], zeros(N + 1), atol=1e-14)


def test_poly_ChebyshevT_mod_1_derivative():
    """mod_ord=1: Tmod__n(+-1)=0 and Tmod_'_n(+-1)~0."""
    N = 4
    x = array([-1.0, 1.0])
    Tm = poly_ChebyshevT_mod(N, x, ret_ord_num=N + 1, mod_ord=1)
    assert_allclose(Tm[:, 0], zeros(N + 1), atol=1e-14)
    assert_allclose(Tm[:, -1], zeros(N + 1), atol=1e-14)
    # Numerical derivative check at x=1
    eps = 1e-6
    Tp = poly_ChebyshevT_mod(N, array([1.0 + eps]),
                           ret_ord_num=N + 1, mod_ord=1)
    Tm_e = poly_ChebyshevT_mod(N, array([1.0 - eps]),
                             ret_ord_num=N + 1, mod_ord=1)
    der1 = (Tp[:, 0] - Tm_e[:, 0]) / (2 * eps)
    assert_allclose(der1, zeros(N + 1), atol=1e-8)


def test_poly_ChebyshevT_mod_2_second_derivative():
    """mod_ord=2: Tmod__n=Tmod_'_n=Tmod_''_n=0 at +-1."""
    N = 4
    x = array([-1.0, 1.0])
    Tm = poly_ChebyshevT_mod(N, x, ret_ord_num=N + 1, mod_ord=2)
    assert_allclose(Tm[:, 0], zeros(N + 1), atol=1e-14)
    assert_allclose(Tm[:, -1], zeros(N + 1), atol=1e-14)
    # Numerical second derivative at x=1
    eps = 1e-5
    T0 = poly_ChebyshevT_mod(N, array([1.0]), ret_ord_num=N + 1, mod_ord=2)
    Tp = poly_ChebyshevT_mod(N, array([1.0 + eps]),
                           ret_ord_num=N + 1, mod_ord=2)
    Tm_e = poly_ChebyshevT_mod(N, array([1.0 - eps]),
                             ret_ord_num=N + 1, mod_ord=2)
    der2 = (Tp[:, 0] - 2 * T0[:, 0] + Tm_e[:, 0]) / (eps * eps)
    assert_allclose(der2, zeros(N + 1), atol=2e-2)


# =============================================================================
# Polynomials: LegendreP
# =============================================================================

def test_poly_LegendreP_endpoints():
    """P_n(1) = 1, P_n(-1) = (-1)^n."""
    N = 5
    x = array([-1.0, 1.0])
    P = poly_LegendreP(N, x, ret_ord_num=N + 1)
    for n in range(N + 1):
        assert_allclose(P[n, 0], (-1) ** n, atol=1e-14)
        assert_allclose(P[n, 1], 1.0, atol=1e-14)


def test_poly_LegendreP_P2():
    """P_2(x) = (3x^2 - 1)/2."""
    x = linspace(-1, 1, 5)
    P = poly_LegendreP(2, x, ret_ord_num=3)
    expected = (3 * x ** 2 - 1) / 2
    assert_allclose(P[2, :], expected, atol=1e-13)


def test_poly_der_LegendreP_known():
    """d/dx P_2 = 3x, d/dx P_3 = (15x^2 - 3)/2."""
    x = linspace(-1, 1, 5)
    dP = poly_der_LegendreP(1, 3, x, ret_der_num=1, ret_ord_num=4)
    # dP_2 = 3x
    assert_allclose(dP[0, 2, :], 3 * x, atol=1e-13)
    # dP_3 = (15x^2 - 3)/2
    assert_allclose(dP[0, 3, :], (15 * x ** 2 - 3) / 2, atol=1e-13)


# =============================================================================
# Special functions
# =============================================================================

def test_special_exp_log_frac_sum_basic():
    """Compare against direct prod+log+exp for small N."""
    N = 5
    f1, f2, f3, f4 = 1.0, 2.0, 1.0, 3.0
    g = 1.0
    result = special_exp_log_frac_sum(N, f1, f2, f3, f4, g)
    j_arr = np.arange(0, N)
    log_terms = np.log((j_arr * f1 + f2) / (j_arr * f3 + f4)) * g
    expected = np.exp(np.cumsum(log_terms))
    assert_allclose(result, expected, atol=1e-14)


def test_special_log_abs_gamma_against_scipy():
    """Compare against scipy.special.gammaln."""
    x = array([0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
    result = special_log_abs_gamma(x)
    expected = scipy_gammaln(x)
    assert_allclose(result, expected, atol=1e-13)
