"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.spectral: polynomial basis construction.
"""

import numpy as np
from numpy import (cos, dot, exp, linspace)
from numpy.testing import assert_allclose

from pynalgo.spectral import (
    poly_basis_mat_ChebyshevT, poly_basis_mat_trig, poly_basis_mat_trig_exp
)
from pynalgo import array_dot_2d_at_axis


def test_poly_basis_mat_ChebyshevT_roundtrip():
    """Extract coeffs, reconstruct, verify against original function."""
    N = 16
    gr = linspace(-1, 1, N + 1)  # doesn't matter, basis creates its own grid
    # Use actual Chebyshev grid for the function sampling
    from pynalgo import grid_ChebyshevT
    gr = grid_ChebyshevT(N, variety="GL")
    fun = cos(3 * np.arccos(gr))  # T_3(gr) - exact Chebyshev polynomial
    wbas = poly_basis_mat_ChebyshevT(N, variety="GL", quad_weight=True)
    bas = poly_basis_mat_ChebyshevT(N, variety="GL", quad_weight=False)
    # Extract coefficients
    coeffs = array_dot_2d_at_axis(wbas, fun)
    # Reconstruct
    rec = dot(bas.T, coeffs)
    assert_allclose(rec, fun, atol=1e-12)


def test_poly_basis_mat_trig_cos_roundtrip():
    """Roundtrip for cosine basis on H1_C grid."""
    N = 16
    from pynalgo import grid_Fourier
    gr = grid_Fourier(N, variety="H1_C")
    fun = cos(4 * gr)  # exact cos(4x)
    wbas = poly_basis_mat_trig(N, variety="H1_C_cos", quad_weight=True)
    bas = poly_basis_mat_trig(N, variety="H1_C_cos", quad_weight=False)
    coeffs = array_dot_2d_at_axis(wbas, fun)
    rec = dot(bas.T, coeffs)
    assert_allclose(rec, fun, atol=1e-12)


def test_poly_basis_mat_trig_exp_roundtrip():
    """Roundtrip for complex exponential basis on S1 grid."""
    N = 16
    from pynalgo import grid_Fourier
    gr = grid_Fourier(N, variety="S1_I")
    fun = exp(2j * gr)  # exact exp(2i*x)
    wbas = poly_basis_mat_trig_exp(N, variety="S1_I", quad_weight=True)
    bas = poly_basis_mat_trig_exp(N, variety="S1_I", quad_weight=False)
    coeffs = array_dot_2d_at_axis(wbas, fun)
    rec = dot(bas.T, coeffs)
    assert_allclose(rec, fun, atol=1e-12)
