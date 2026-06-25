"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test pynalgo.number_theory: prime generation,
           factorization, arithmetic.
"""

import numpy as np
from numpy.testing import assert_array_equal

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

# =============================================================================
# generate_primes
# =============================================================================

def test_generate_primes_small():
  assert_array_equal(generate_primes(0), np.array([], dtype=np.int64))
  assert_array_equal(generate_primes(1), np.array([], dtype=np.int64))
  assert_array_equal(generate_primes(2), np.array([2], dtype=np.int64))
  assert_array_equal(generate_primes(10),
                     np.array([2, 3, 5, 7], dtype=np.int64))


def test_generate_primes_count():
  """First 25 primes: number of primes <= 100 is 25."""
  assert generate_primes(100).size == 25


PRIMES_UNDER_30 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29], dtype=np.int64)


def test_generate_primes_under_30():
  assert_array_equal(generate_primes(30), PRIMES_UNDER_30)


def test_generate_primes_under_100():
  p = generate_primes(100)
  assert p[0] == 2
  assert p[-1] == 97
  # All returned values are odd except 2
  assert np.all((p[1:] % 2) == 1)


# =============================================================================
# is_prime
# =============================================================================

def test_is_prime_basic():
  assert is_prime(2) is True
  assert is_prime(3) is True
  assert is_prime(4) is False
  assert is_prime(17) is True
  assert is_prime(18) is False
  assert is_prime(0) is False
  assert is_prime(1) is False


def test_is_prime_large():
  assert is_prime(7919) is True   # 1000th prime
  assert is_prime(7920) is False


# =============================================================================
# get_prime_factors
# =============================================================================

def test_get_prime_factors_edge():
  # n < 2: empty
  assert_array_equal(get_prime_factors(1), np.zeros((0, 2), dtype=np.int64))


def test_get_prime_factors_prime():
  result = get_prime_factors(17)
  assert_array_equal(result, np.array([[17, 1]], dtype=np.int64))


def test_get_prime_factors_composite():
  result = get_prime_factors(84)
  expected = np.array([[2, 2], [3, 1], [7, 1]], dtype=np.int64)
  assert_array_equal(result, expected)


def test_get_prime_factors_power_of_two():
  result = get_prime_factors(64)
  assert_array_equal(result, np.array([[2, 6]], dtype=np.int64))


def test_get_prime_factors_product():
  """Verify prod(p_i^e_i) = N."""
  for N in [12, 30, 60, 84, 100, 128, 256, 1000]:
    factors = get_prime_factors(N)
    prod = np.int64(1)
    for i in range(factors.shape[0]):
      prod *= factors[i, 0] ** factors[i, 1]
    assert prod == N, f"N={N}, reconstruction={prod}"


# =============================================================================
# gcd
# =============================================================================

def test_gcd_basic():
  assert gcd(48, 18) == 6
  assert gcd(17, 13) == 1
  assert gcd(0, 5) == 5
  assert gcd(5, 0) == 5
  assert gcd(1, 1) == 1


def test_gcd_large():
  assert gcd(1071, 462) == 21


# =============================================================================
# lcm
# =============================================================================

def test_lcm_basic():
  assert lcm(12, 18) == 36
  assert lcm(17, 13) == 221
  assert lcm(8, 9) == 72
  assert lcm(1, 5) == 5


# =============================================================================
# solve_lin_diophantine
# =============================================================================

def test_solve_lin_diophantine():
  result = solve_lin_diophantine(48, 18)
  x, y, g = result[0], result[1], result[2]
  assert 48 * x + 18 * y == g
  assert g == 6


def test_solve_lin_diophantine_coprime():
  result = solve_lin_diophantine(17, 13)
  x, y, g = result[0], result[1], result[2]
  assert 17 * x + 13 * y == g
  assert g == 1


# =============================================================================
# next_pow2
# =============================================================================

def test_next_pow2():
  assert next_pow2(1) >= 1
  assert 2 ** next_pow2(8) >= 2 * 8 + 1
  assert 2 ** next_pow2(15) >= 2 * 15 + 1
  # Verify minimality: p-1 does not satisfy
  p = next_pow2(8)
  assert 2 ** (p - 1) < 2 * 8 + 1


# =============================================================================
# get_num_divisors
# =============================================================================

def test_get_num_divisors():
  assert get_num_divisors(1) == 1
  assert get_num_divisors(2) == 2       # 1, 2
  assert get_num_divisors(6) == 4       # 1, 2, 3, 6
  assert get_num_divisors(12) == 6      # 1, 2, 3, 4, 6, 12
  assert get_num_divisors(28) == 6      # 1, 2, 4, 7, 14, 28


# =============================================================================
# prime_sqrt_decomp
# =============================================================================

def test_prime_sqrt_decomp_trivial():
  assert prime_sqrt_decomp(0) == 1
  assert prime_sqrt_decomp(1) == 1
  assert prime_sqrt_decomp(2) == 1


def test_prime_sqrt_decomp():
  # N=36: sqrt=6, factors 2^2*3^2 -> 6 = 2*3 is compatible
  assert prime_sqrt_decomp(36) == 6
  # N=100: sqrt=10, factors 2^2*5^2 -> 10 = 2*5
  assert prime_sqrt_decomp(100) == 10
  # N=256: sqrt=16, factors 2^8 -> 16 = 2^4
  assert prime_sqrt_decomp(256) == 16


def test_prime_sqrt_decomp_non_square():
  assert prime_sqrt_decomp(30) == 5
  assert prime_sqrt_decomp(60) == 6
  assert prime_sqrt_decomp(84) == 7


# =============================================================================
# prime_power_arrs_to_list
# =============================================================================

def test_prime_power_arrs_to_list():
  primes  = np.array([2, 3, 7], dtype=np.int64)
  powers  = np.array([2, 1, 1], dtype=np.int64)
  result = prime_power_arrs_to_list(primes, powers)
  expected = np.array([2, 2, 3, 7], dtype=np.int64)
  assert_array_equal(result, expected)


def test_prime_power_arrs_to_list_single():
  primes = np.array([5], dtype=np.int64)
  powers = np.array([3], dtype=np.int64)
  result = prime_power_arrs_to_list(primes, powers)
  expected = np.array([5, 5, 5], dtype=np.int64)
  assert_array_equal(result, expected)
