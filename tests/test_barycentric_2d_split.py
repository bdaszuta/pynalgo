"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.resample.barycentric_2d_split:
           ADI-like dimension-split interpolation.
"""
import numpy as np
from numpy import linspace, meshgrid
from numpy.testing import assert_allclose

from pynalgo.resample import interp_barycentric_2d_split


def test_separable_exact():
    """Separable function f(x,y)=g(x)*h(y): exact reconstruction."""
    x_src = linspace(-1, 1, 30)
    y_src = linspace(-1, 1, 25)
    X_src, Y_src = meshgrid(x_src, y_src, indexing='ij')
    F_src = np.sin(3 * X_src) * np.cos(2 * Y_src)

    x_tgt = linspace(-0.9, 0.9, 20)
    y_tgt = linspace(-0.9, 0.9, 15)

    F_tgt = interp_barycentric_2d_split(
        x_src, y_src, F_src, x_tgt, y_tgt)

    X_tgt, Y_tgt = meshgrid(x_tgt, y_tgt, indexing='ij')
    F_exact = np.sin(3 * X_tgt) * np.cos(2 * Y_tgt)
    assert_allclose(F_tgt, F_exact, atol=1e-4)


def test_polynomial_exact():
    """Polynomial f(x,y)=x^2*y^3: exact reconstruction."""
    x_src = linspace(-1, 1, 20)
    y_src = linspace(-1, 1, 18)
    X_src, Y_src = meshgrid(x_src, y_src, indexing='ij')
    F_src = X_src ** 2 * Y_src ** 3

    x_tgt = linspace(-0.8, 0.8, 10)
    y_tgt = linspace(-0.8, 0.8, 8)

    F_tgt = interp_barycentric_2d_split(
        x_src, y_src, F_src, x_tgt, y_tgt)

    X_tgt, Y_tgt = meshgrid(x_tgt, y_tgt, indexing='ij')
    F_exact = X_tgt ** 2 * Y_tgt ** 3
    assert_allclose(F_tgt, F_exact, atol=1e-4)


def test_shape():
    """Output has correct shape (n_tgt_x, n_tgt_y)."""
    x_src = linspace(-1, 1, 20)
    y_src = linspace(-1, 1, 15)
    X_src, Y_src = meshgrid(x_src, y_src, indexing='ij')
    F_src = X_src * Y_src

    x_tgt = linspace(-0.5, 0.5, 12)
    y_tgt = linspace(-0.5, 0.5, 8)

    F_tgt = interp_barycentric_2d_split(
        x_src, y_src, F_src, x_tgt, y_tgt)
    assert F_tgt.shape == (12, 8)


#
# :D
#
