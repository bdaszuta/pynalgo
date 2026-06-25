"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.common_tools: primitives,
           numba utilities, numpy array utilities.
"""

from numpy import (array, arange, diag, dot, float64, int64, ones,
                   complex128, linspace)
from numpy.testing import assert_allclose

from pynalgo.common_tools import (
    is_even, is_odd, idx_interchange,
    array_common_type, array_extend_mirror, array_extend_uniform_range,
    array_dense_to_banded, array_dense_from_band_uniform,
    array_dense_from_bands, array_identity_mask, array_pad_biased,
    array_roll_row_to_edges, array_dot_band_uniform, array_dot_bands,
    array_dot_2d_at_axis,
    arg_extremum_interval, arg_extremum_disc,
    ndarray_get_sorted_argmin, interval_is_intersection_empty,
    scalar_data_deduplicate_sort, ndarray_dim_order_exchange,
    map_window_exp
)


# =============================================================================
# Primitives
# =============================================================================

def test_is_even():
    for v in [0, 2, 4, -2, -10, 100]:
        assert is_even(v) is True
    for v in [1, 3, 5, -1, -3, 101]:
        assert is_even(v) is False


def test_is_odd():
    for v in [1, 3, 5, -1, -3, 101]:
        assert is_odd(v) is True
    for v in [0, 2, 4, -2, -10, 100]:
        assert is_odd(v) is False


# =============================================================================
# numba utilities
# =============================================================================

def test_idx_interchange_copy():
    """Copy mode: original unchanged, returned array has swapped elements."""
    orig = (0, 1, 2, 3)
    result = idx_interchange(orig, 0, 3)
    assert array(orig).tolist() == [0, 1, 2, 3]
    assert result.tolist() == [3, 1, 2, 0]


def test_idx_interchange_inplace():
    """Inplace mode: original modified, returned array is original."""
    arr = array([10, 20, 30, 40])
    result = idx_interchange(arr, 1, 2, inplace=True)
    assert result is arr
    assert arr.tolist() == [10, 30, 20, 40]


# =============================================================================
# array_common_type
# =============================================================================

def test_array_common_type_int_float():
    A = array([1, 2, 3], dtype=int64)
    B = array([2.2, 0.0], dtype=float64)
    assert array_common_type(A, B) == float64


def test_array_common_type_float_complex():
    A = array([1.0, 2.0], dtype=float64)
    B = array([1j, 2j], dtype=complex128)
    assert array_common_type(A, B) == complex128


def test_array_common_type_same():
    A = array([1.0, 2.0], dtype=float64)
    B = array([3.0, 4.0], dtype=float64)
    assert array_common_type(A, B) == float64


# =============================================================================
# array_extend_mirror
# =============================================================================

def test_array_extend_mirror_axis0():
    arr = array([[1, 2, 3],
                 [3, 2, 1],
                 [4, 5, 6]])
    result = array_extend_mirror(arr, axis=0)
    expected = array([[1, 2, 3],
                      [3, 2, 1],
                      [4, 5, 6],
                      [3, 2, 1],
                      [1, 2, 3]])
    assert_allclose(result, expected)


def test_array_extend_mirror_axis1():
    arr = array([[1, 2, 3],
                 [3, 2, 1],
                 [4, 5, 6]])
    result = array_extend_mirror(arr, axis=1)
    expected = array([[1, 2, 3, 2, 1],
                      [3, 2, 1, 2, 3],
                      [4, 5, 6, 5, 4]])
    assert_allclose(result, expected)


# =============================================================================
# array_extend_uniform_range
# =============================================================================

def test_array_extend_uniform_range_both():
    s = linspace(-2, 1, num=3)
    result = array_extend_uniform_range(s, NGHOST=2)
    expected = array([-5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0])
    assert_allclose(result, expected)


def test_array_extend_uniform_range_left_only():
    s = linspace(-2, 1, num=3)
    result = array_extend_uniform_range(s, NGHOST=2, extend_right=False)
    expected = array([-5.0, -3.5, -2.0, -0.5, 1.0])
    assert_allclose(result, expected)


def test_array_extend_uniform_range_right_only():
    s = linspace(-2, 1, num=3)
    result = array_extend_uniform_range(s, NGHOST=2, extend_left=False)
    expected = array([-2.0, -0.5, 1.0, 2.5, 4.0])
    assert_allclose(result, expected)


# =============================================================================
# array_dense_to_banded
# =============================================================================

def test_array_dense_to_banded_tridiagonal():
    snake = arange(1, 26).reshape((5, 5))
    result = array_dense_to_banded(snake, width=3)
    expected = array([[0, 1, 2],
                      [6, 7, 8],
                      [12, 13, 14],
                      [18, 19, 20],
                      [24, 25, 0]])
    assert_allclose(result, expected)


def test_array_dense_to_banded_width1():
    """width=1 extracts only the main diagonal."""
    A = arange(1, 10).reshape((3, 3))
    result = array_dense_to_banded(A, width=1)
    assert result.shape == (3, 1)
    assert result[:, 0].tolist() == [1, 5, 9]


# =============================================================================
# array_dense_from_band_uniform
# =============================================================================

def test_array_dense_from_band_uniform_basic():
    banded = array([1, 2, 3])
    result = array_dense_from_band_uniform(banded, 4)
    expected = array([[2., 3., 0., 0.],
                      [1., 2., 3., 0.],
                      [0., 1., 2., 3.],
                      [0., 0., 1., 2.]])
    assert_allclose(result, expected)


def test_array_dense_from_band_uniform_roundtrip():
    """Uniform tridiagonal: sub=1, main=2, super=3.
    Interior band row = [1,2,3]."""
    N = 6
    A = (diag(2 * ones(N))
         + diag(1 * ones(N - 1), k=-1)
         + diag(3 * ones(N - 1), k=1))
    band_vec = array([1.0, 2.0, 3.0])
    result = array_dense_from_band_uniform(band_vec, N)
    assert_allclose(result, A)


# =============================================================================
# array_dense_from_bands
# =============================================================================

def test_array_dense_from_bands_basic():
    bands = array([[0, 2, 3],
                   [1, 4, 1],
                   [3, 2, 3],
                   [3, 2, 0]])
    N = 6
    result = array_dense_from_bands(bands, N, idx_center_fill=1)
    expected = array([[2, 3, 0, 0, 0, 0],
                      [1, 4, 1, 0, 0, 0],
                      [0, 1, 4, 1, 0, 0],
                      [0, 0, 1, 4, 1, 0],
                      [0, 0, 0, 3, 2, 3],
                      [0, 0, 0, 0, 3, 2]])
    assert_allclose(result, expected)


# =============================================================================
# array_identity_mask
# =============================================================================

def test_array_identity_mask_interior():
    result = array_identity_mask(1, 5, 6)
    expected = array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0]])
    assert_allclose(result, expected)


def test_array_identity_mask_complement():
    result = array_identity_mask(1, 5, 6, is_complement=True)
    expected = array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
    assert_allclose(result, expected)


# =============================================================================
# array_pad_biased
# =============================================================================

def test_array_pad_biased_left():
    """c_node=0 means left edge is center -> pads right."""
    result = array_pad_biased(array([1, 2, 3]), c_node=0)
    expected = array([0, 0, 1, 2, 3])
    assert_allclose(result, expected)


def test_array_pad_biased_center():
    """c_node=1 is exact center of 3-element array -> no pad."""
    result = array_pad_biased(array([1, 2, 3]), c_node=1)
    expected = array([1, 2, 3])
    assert_allclose(result, expected)


def test_array_pad_biased_right():
    """c_node=2 means right edge is center -> pads left."""
    result = array_pad_biased(array([1, 2, 3]), c_node=2)
    expected = array([1, 2, 3, 0, 0])
    assert_allclose(result, expected)


# =============================================================================
# array_roll_row_to_edges
# =============================================================================

def test_array_roll_row_to_edges_basic():
    A = array([[1, 2, 3, 0, 0, 0, 0],
               [-2, 1, 2, 3, 0, 0, 0],
               [-3, -2, 1, 2, 3, 0, 0],
               [0, -3, -2, 1, 2, 3, 0],
               [0, 0, -3, -2, 1, 2, 3],
               [0, 0, 0, -3, -2, 1, 2],
               [0, 0, 0, 0, -3, -2, 1]])
    array_roll_row_to_edges(A, 2, num_skipped=1)
    expected = array([[1, 2, 3, 0, -3, -2, 0],
                      [-2, 1, 2, 3, 0, -3, 0],
                      [-3, -2, 1, 2, 3, 0, 0],
                      [0, -3, -2, 1, 2, 3, 0],
                      [0, 0, -3, -2, 1, 2, 3],
                      [0, 3, 0, -3, -2, 1, 2],
                      [0, 2, 3, 0, -3, -2, 1]])
    assert_allclose(A, expected)


# =============================================================================
# array_dot_band_uniform
# =============================================================================

def test_array_dot_band_uniform_axis0():
    band = array([2, 4.1, 2, 3, 8.1])
    N = 8
    x = arange(2, 2 + N).reshape((N, 1))
    result = array_dot_band_uniform(band, x)
    expected_dense = array_dense_from_band_uniform(band, N)
    expected = dot(expected_dense, x)
    assert_allclose(result, expected)


def test_array_dot_band_uniform_axis1():
    band = array([2, 4.1, 2, 3, 8.1])
    N = 8  # rows of xy
    M = 6  # cols of xy (this is the band matrix dimension for axis=1)
    x = arange(2, 2 + N).reshape((N, 1))
    y = arange(1, 1 + M).reshape((1, M))
    xy = x + y
    result = array_dot_band_uniform(band, xy, axis=1)
    expected_dense = array_dense_from_band_uniform(band, M)
    # axis=1: band matrix of size M dots each row of xy (shape N x M)
    # NumPy equivalent:
    #   dot(band_MxM, xy.T).T = (M,M) @ (M,N) -> (M,N).T -> (N,M)
    expected = dot(expected_dense, xy.T).T
    assert_allclose(result, expected)


# =============================================================================
# array_dot_bands
# =============================================================================

def test_array_dot_bands_vs_dense():
    """Uniform pentadiagonal banded dot should match dense dot."""
    N = 8
    band_vec = array([-1.0, 2.0, 4.0, 3.0, 1.0])
    # tile band vector N times to get (N, 5) bands
    bands = band_vec[None, :] * ones((N, 1))
    dense = array_dense_from_band_uniform(band_vec, N)
    x = arange(1, N + 1, dtype=float64).reshape((N, 1))
    result = array_dot_bands(bands, x)
    expected = dot(dense, x)
    assert_allclose(result, expected)


# =============================================================================
# array_dot_2d_at_axis
# =============================================================================

def test_array_dot_2d_at_axis_axis0():
    """For 1D fcn, axis=0 does dot(mat, fcn) = mat@fcn, fcn matches cols."""
    mat = arange(1, 13).reshape((3, 4)).astype(float64)
    vec = array([0.5, 1.0, 2.0, 3.0])
    result = array_dot_2d_at_axis(mat, vec, axis=0)
    expected = dot(mat, vec)
    assert_allclose(result, expected)


def test_array_dot_2d_at_axis_axis1():
    """For 1D fcn, axis=1 also does dot(mat, fcn) (identical to axis=0)."""
    mat = arange(1, 13).reshape((3, 4)).astype(float64)
    vec = array([0.5, 1.0, 2.0, 3.0])
    result = array_dot_2d_at_axis(mat, vec, axis=1)
    expected = dot(mat, vec)
    assert_allclose(result, expected)


# =============================================================================
# arg_extremum_interval
# =============================================================================

def test_arg_extremum_interval_max():
    """Find argmax within interval on a known 1D array."""
    x_I = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    F_I = array([5.0, 10.0, 2.0, 8.0, 3.0, 7.0])
    # Interval |x - 2.0| < 1.5  =>  x in (0.5, 3.5)  =>  indices 1,2,3
    # F values at those indices: 10, 2, 8  -> max is 10 at index 1
    ix = arg_extremum_interval(x_I, F_I, x_0=2.0, r_0=1.5, extrema_type=+1)
    assert ix == 1


def test_arg_extremum_interval_min():
    """Find argmin within interval on a known 1D array."""
    x_I = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    F_I = array([5.0, 10.0, 2.0, 8.0, 3.0, 7.0])
    # Interval |x - 2.0| < 1.5  =>  x in (0.5, 3.5)  =>  indices 1,2,3
    # F values: 10, 2, 8  -> min is 2 at index 2
    ix = arg_extremum_interval(x_I, F_I, x_0=2.0, r_0=1.5, extrema_type=-1)
    assert ix == 2


def test_arg_extremum_interval_empty_returns_first():
    """If no point lies in the interval, returns index 0."""
    x_I = array([0.0, 1.0, 2.0])
    F_I = array([5.0, 3.0, 1.0])
    ix = arg_extremum_interval(x_I, F_I, x_0=10.0, r_0=0.5)
    assert ix == 0


# =============================================================================
# arg_extremum_disc
# =============================================================================

def test_arg_extremum_disc_max():
    """Find argmax within disc on a 2D grid."""
    x_I = array([0.0, 1.0, 2.0, 3.0])
    y_I = array([0.0, 1.0, 2.0, 3.0])
    F_I = ones((4, 4))
    F_I[2, 1] = 100.0   # peak at (x=2, y=1)
    # Disc centred at (1.5, 1.5), radius 1.5:
    #   (x-1.5)^2 + (y-1.5)^2 < 2.25
    # Points inside: (1,1), (1,2), (2,1), (2,2)
    ix, jx = arg_extremum_disc(x_I, y_I, F_I, x_0=1.5, y_0=1.5, r_0=1.5,
                               extrema_type=+1)
    assert ix == 2
    assert jx == 1


def test_arg_extremum_disc_min():
    """Find argmin within disc on a 2D grid."""
    x_I = array([0.0, 1.0, 2.0, 3.0])
    y_I = array([0.0, 1.0, 2.0, 3.0])
    F_I = 10.0 * ones((4, 4))
    F_I[1, 2] = 0.0    # valley at (x=1, y=2)
    # Same disc: centre (1.5, 1.5), radius 1.5
    ix, jx = arg_extremum_disc(x_I, y_I, F_I, x_0=1.5, y_0=1.5, r_0=1.5,
                               extrema_type=-1)
    assert ix == 1
    assert jx == 2


def test_arg_extremum_disc_empty_returns_origin():
    """If no grid point lies inside the disc, returns (0, 0)."""
    x_I = array([0.0, 1.0])
    y_I = array([0.0, 1.0])
    F_I = ones((2, 2))
    ix, jx = arg_extremum_disc(x_I, y_I, F_I, x_0=10.0, y_0=10.0, r_0=0.5)
    assert ix == 0
    assert jx == 0


# =============================================================================
# ndarray_get_sorted_argmin
# =============================================================================

def test_ndarray_get_sorted_argmin_val_inside():
    """Find index nearest to a value inside the array range."""
    arr = array([1.0, 3.0, 5.0, 7.0, 9.0])
    # val=6: |5-6|=1, |7-6|=1; first minimum wins at index 2
    ix = ndarray_get_sorted_argmin(6.0, arr)
    assert ix == 2


def test_ndarray_get_sorted_argmin_val_below():
    """Value below the sorted array -- first element is closest."""
    arr = array([1.0, 3.0, 5.0])
    ix = ndarray_get_sorted_argmin(0.0, arr)
    assert ix == 0


def test_ndarray_get_sorted_argmin_val_above():
    """Value above the sorted array -- last element is closest."""
    arr = array([1.0, 3.0, 5.0])
    ix = ndarray_get_sorted_argmin(100.0, arr)
    assert ix == 2


def test_ndarray_get_sorted_argmin_exact_match():
    """Exact match returns its index."""
    arr = array([1.0, 3.0, 5.0, 7.0])
    ix = ndarray_get_sorted_argmin(5.0, arr)
    assert ix == 2


# =============================================================================
# interval_is_intersection_empty
# =============================================================================

def test_interval_intersection_overlapping():
    """Overlapping intervals: [0,5] intersect [3,8] is non-empty."""
    result = interval_is_intersection_empty(0, 5, 3, 8)
    assert result is False


def test_interval_intersection_first_before_second():
    """[0,2] intersect [5,8] is empty (first entirely before second)."""
    result = interval_is_intersection_empty(0, 2, 5, 8)
    assert result is True


def test_interval_intersection_first_after_second():
    """[5,8] intersect [0,2] is empty (first entirely after second)."""
    result = interval_is_intersection_empty(5, 8, 0, 2)
    assert result is True


def test_interval_intersection_touching():
    """Touching at a single point is not considered empty."""
    result = interval_is_intersection_empty(0, 5, 5, 8)
    assert result is False


def test_interval_intersection_contained():
    """Interval [2,4] inside [0,6] is non-empty."""
    result = interval_is_intersection_empty(2, 4, 0, 6)
    assert result is False


# =============================================================================
# scalar_data_deduplicate_sort
# =============================================================================

def test_scalar_data_deduplicate_sort_basic():
    """2D data with duplicate entries in column 0: sort + dedup."""
    data = array([[2.0, 10.0],
                  [1.0, 20.0],
                  [2.0, 30.0],
                  [3.0, 40.0],
                  [1.0, 50.0]])
    result = scalar_data_deduplicate_sort(data, column=0)
    expected = array([[1.0, 20.0],
                      [2.0, 10.0],
                      [3.0, 40.0]])
    assert_allclose(result, expected)


def test_scalar_data_deduplicate_sort_no_duplicates():
    """If no duplicates, data is just sorted by column."""
    data = array([[3.0, 30.0],
                  [1.0, 10.0],
                  [2.0, 20.0]])
    result = scalar_data_deduplicate_sort(data, column=0)
    expected = array([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]])
    assert_allclose(result, expected)


def test_scalar_data_deduplicate_sort_with_indices():
    """Return indices alongside deduplicated data."""
    data = array([[2.0, 10.0],
                  [1.0, 20.0],
                  [3.0, 40.0]])
    result, indices = scalar_data_deduplicate_sort(data, column=0,
                                                   return_indices=True)
    expected_data = array([[1.0, 20.0],
                           [2.0, 10.0],
                           [3.0, 40.0]])
    assert_allclose(result, expected_data)
    assert indices.tolist() == [1, 0, 2]


# =============================================================================
# ndarray_dim_order_exchange
# =============================================================================

def test_ndarray_dim_order_exchange_skip_none():
    """num_skip_dim=0 reverses all dimensions (C<->Fortran swap)."""
    arr = ones((2, 3, 4, 5))
    result = ndarray_dim_order_exchange(arr, num_skip_dim=0)
    assert result.shape == (5, 4, 3, 2)


def test_ndarray_dim_order_exchange_skip_one():
    """num_skip_dim=1 preserves first axis, reverses the rest."""
    arr = ones((2, 3, 4, 5))
    result = ndarray_dim_order_exchange(arr, num_skip_dim=1)
    assert result.shape == (2, 5, 4, 3)


def test_ndarray_dim_order_exchange_skip_two():
    """num_skip_dim=2 preserves first two axes, reverses the rest."""
    arr = ones((2, 3, 4, 5, 6))
    result = ndarray_dim_order_exchange(arr, num_skip_dim=2)
    assert result.shape == (2, 3, 6, 5, 4)


def test_ndarray_dim_order_exchange_single_trailing():
    """When only one trailing dimension, its position is unchanged."""
    arr = ones((2, 3, 4))
    result = ndarray_dim_order_exchange(arr, num_skip_dim=2)
    assert result.shape == (2, 3, 4)


# =============================================================================
# map_window_exp
# =============================================================================

def test_map_window_exp_delta_zero():
    """delta=0: all points in [0,1] return 1, except x=1 returns 0."""
    x = linspace(0.0, 1.0, num=11)
    w = map_window_exp(x, delta=0.0)
    # With delta=0, piecewise's last-match semantics: at x=0,
    # condition (x==0) matches first, then (x>=0 & x<=1) matches later
    # and takes precedence, yielding 1.0.  At x=1, (x==1) is last -> 0.0.
    expected = ones(11)
    expected[-1] = 0.0
    assert_allclose(w, expected)


def test_map_window_exp_delta_nonzero_transition():
    """delta=0.1: flat interior is 1.0, boundaries go smoothly to 0."""
    x = linspace(0.0, 1.0, num=21)
    w = map_window_exp(x, delta=0.1)
    # Interior (far from boundaries): should be 1.0
    interior_slice = slice(4, 17)  # x in [0.2, 0.8], well inside
    assert_allclose(w[interior_slice], ones(13), atol=1e-14)
    # Boundaries: value should be 0.0 exactly at x=0 and x=1
    assert_allclose(w[0], 0.0)
    assert_allclose(w[-1], 0.0)
    # Transition region: values between 0 and 1 (not leaning on plateau)
    assert 0.0 < w[1] < 1.0
    assert 0.0 < w[-2] < 1.0
    # Symmetry: w(x) == w(1 - x)
    assert_allclose(w[1], w[-2])


def test_map_window_exp_delta_nonzero_values():
    """Explicit values for delta=0.1 at a few sample points."""
    x = array([0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0])
    w = map_window_exp(x, delta=0.1)
    # x=0 and x=1 map to 0 exactly
    assert_allclose(w[0], 0.0)
    assert_allclose(w[-1], 0.0)
    # x=0.05 is in the left transition region: 0 < w < 1
    assert 0.0 < w[1] < 1.0
    # x=0.15, 0.5, 0.85 are in the flat interior: w ~ 1
    assert_allclose(w[2:5], ones(3), atol=1e-14)
    # x=0.95 is in the right transition region: 0 < w < 1
    assert 0.0 < w[5] < 1.0
