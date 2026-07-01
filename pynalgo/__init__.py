"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Python numerical algorithms library: spectral methods, grids,
           differentiation, integration, interpolation, rational
           approximation (AAA), root-finding, FFT, DCT, number theory,
           linear algebra.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pynalgo")
except PackageNotFoundError:
    __version__ = "0.1.0"

from pynalgo.common_tools import (
  # utilities_primitives
  is_even,
  is_odd,
  # utilities_numba
  idx_interchange,
  # utilities_numpy
  array_common_type,
  array_dense_from_band_uniform,
  array_dense_from_bands,
  array_dense_to_banded,
  array_extend_mirror,
  array_extend_uniform_range,
  array_identity_mask,
  array_pad_biased,
  array_roll_row_to_edges,
  array_dot_band_uniform,
  array_dot_bands,
  array_dot_2d_at_axis,
  arg_extremum_interval,
  arg_extremum_disc,
  ndarray_get_sorted_argmin,
  interval_is_intersection_empty,
  scalar_data_deduplicate_sort,
  ndarray_dim_order_exchange,
  map_window_exp)

from pynalgo.differentiation import (
  # nodal_spectral
  diff_mat_nodal_ChebyshevT,
  diff_mat_nodal_Fourier,
  diff_mat_nodal_JacobiP,
)

from pynalgo.integration import (
  # quadrature
  quad_ChebyshevT,
  quad_Clenshaw_Curtis,
  quad_Fourier,
  quad_JacobiP,
  quad_LegendreP,
)

from pynalgo.linear_algebra import (
  # solvers
  solver_tridiagonal,
  solver_pentadiagonal,
  solver_pentadiagonalF,
  solver_pentadiagonalFS,
)

from pynalgo.rational import (
  # aaa
  aaa,
  aaa_real,
  eval_rat,
  poles_residues,
  rat_D1,
  rat_D2,
  rat_D,
  diff_mat_nodal_rat,
)

from pynalgo.resample import (
  interp_barycentric_1d,
  interp_barycentric_1d_generalized,
  interp_nn_1d,
  interp_lagrange_uniform,
  interp_barycentric_nd,
  Lebesgue_func,
  columns_interpolate,
  lerp_1d,
  uniform_filter_1d,
  gaussian_filter_1d,
  columns_smooth,
  extrap_Richardson,
  extrap_Richardson_err,
  interp_Neville,
)

from pynalgo.special_functions import (
  # factorial
  special_exp_log_frac_sum,
  special_log_abs_gamma,
  # grids
  grid_ChebyshevT,
  grid_JacobiP,
  grid_LegendreP,
  grid_Fourier,
  # recursion_polynomial
  poly_JacobiP,
  poly_der_JacobiP,
  poly_ChebyshevT,
  poly_der_ChebyshevT,
  poly_ChebyshevU,
  poly_der_ChebyshevU,
  poly_ChebyshevV,
  poly_der_ChebyshevV,
  poly_ChebyshevW,
  poly_der_ChebyshevW,
  poly_Gegenbauer,
  poly_der_Gegenbauer,
  poly_Ultraspherical,
  poly_der_Ultraspherical,
  poly_ChebyshevT_direct,
  poly_ChebyshevU_direct,
  poly_Laguerre,
  poly_Laguerre_lambda,
  poly_der_Laguerre,
  poly_der_Laguerre_lambda,
  poly_sin,
  poly_cos,
  poly_Hermite_psi,
  poly_Hermite_H,
  poly_der_Hermite_H,
  poly_ChebyshevT_mod,
  poly_LegendreP,
  poly_der_LegendreP,
)

from pynalgo.number_theory import (
  generate_primes,
  is_prime,
  get_prime_factors,
  prime_sqrt_decomp,
  get_num_divisors,
  gcd,
  lcm,
  solve_lin_diophantine,
  next_pow2,
  prime_power_arrs_to_list,
)

from pynalgo.fft import (
  fft,
  ifft,
  build_plan,
)

from pynalgo.root_finding import (
  # chebyshev_proxy
  chebyshev_proxy,
)

from pynalgo.spectral import (
  # bases
  poly_basis_mat_ChebyshevT,
  poly_basis_mat_trig,
  poly_basis_mat_trig_exp,
  # dct
  dct1,
  dct2,
  dct3,
  dct4,
)

#
# :D
#

__all__ = [
  'array_common_type',
  'array_dense_from_band_uniform',
  'array_dense_from_bands',
  'array_dense_to_banded',
  'array_dot_2d_at_axis',
  'array_dot_band_uniform',
  'array_dot_bands',
  'array_extend_mirror',
  'array_extend_uniform_range',
  'array_identity_mask',
  'array_pad_biased',
  'array_roll_row_to_edges',
  'build_plan',
  'chebyshev_proxy',
  'dct1',
  'dct2',
  'dct3',
  'dct4',
  'diff_mat_nodal_ChebyshevT',
  'diff_mat_nodal_Fourier',
  'diff_mat_nodal_JacobiP',
  'fft',
  'gcd',
  'generate_primes',
  'get_num_divisors',
  'get_prime_factors',
  'grid_ChebyshevT',
  'grid_Fourier',
  'grid_JacobiP',
  'grid_LegendreP',
  'idx_interchange',
  'ifft',
  'is_even',
  'is_odd',
  'is_prime',
  'lcm',
  'next_pow2',
  'poly_ChebyshevT',
  'poly_ChebyshevT_direct',
  'poly_ChebyshevT_mod',
  'poly_ChebyshevU',
  'poly_ChebyshevU_direct',
  'poly_ChebyshevV',
  'poly_ChebyshevW',
  'poly_Gegenbauer',
  'poly_Hermite_H',
  'poly_Hermite_psi',
  'poly_JacobiP',
  'poly_Laguerre',
  'poly_Laguerre_lambda',
  'poly_LegendreP',
  'poly_Ultraspherical',
  'poly_basis_mat_ChebyshevT',
  'poly_basis_mat_trig',
  'poly_basis_mat_trig_exp',
  'poly_cos',
  'poly_der_ChebyshevT',
  'poly_der_ChebyshevU',
  'poly_der_ChebyshevV',
  'poly_der_ChebyshevW',
  'poly_der_Gegenbauer',
  'poly_der_Hermite_H',
  'poly_der_JacobiP',
  'poly_der_Laguerre',
  'poly_der_Laguerre_lambda',
  'poly_der_LegendreP',
  'poly_der_Ultraspherical',
  'poly_sin',
  'prime_power_arrs_to_list',
  'prime_sqrt_decomp',
  'quad_ChebyshevT',
  'quad_Clenshaw_Curtis',
  'quad_Fourier',
  'quad_JacobiP',
  'quad_LegendreP',
  'solve_lin_diophantine',
  'solver_tridiagonal',
  'solver_pentadiagonal',
  'solver_pentadiagonalF',
  'solver_pentadiagonalFS',
  'special_exp_log_frac_sum',
  'special_log_abs_gamma',
  # resample
  'interp_barycentric_1d',
  'interp_barycentric_1d_generalized',
  'interp_nn_1d',
  'interp_lagrange_uniform',
  'interp_barycentric_nd',
  'Lebesgue_func',
  'columns_interpolate',
  'columns_smooth',
  'lerp_1d',
  'uniform_filter_1d',
  'gaussian_filter_1d',
  'extrap_Richardson',
  'extrap_Richardson_err',
  'interp_Neville',
  # common_tools additions
  'arg_extremum_interval',
  'arg_extremum_disc',
  'ndarray_get_sorted_argmin',
  'interval_is_intersection_empty',
  'scalar_data_deduplicate_sort',
  'ndarray_dim_order_exchange',
  'map_window_exp',
  # rational
  'aaa',
  'aaa_real',
  'eval_rat',
  'poles_residues',
  'rat_D1',
  'rat_D2',
  'rat_D',
  'diff_mat_nodal_rat',
]

#
# :D
#
