"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Prime number generation, factorization, and number-theoretic
           utilities.

Refs:
  [1] Python Sieve: 48M Primes in 4 sec
      https://www.kaggle.com/chefele/introducing-kaggle-scripts/
      scripts-first-try

  [2] Numerical Recipes
      https://bitbucket.org/jaime/numerical-recipes
"""
from numpy import (array, bool_, concatenate, empty,
                   float64, int64, nonzero, ones,
                   sqrt, zeros)

from pynalgo.common_tools import (JIT, NDArray, TYPE_CHECKING)

###############################################################################
# Internal: Eratosthenes sieve (odds only)
###############################################################################
def _sieve_primes(n : int) -> NDArray[int64]:
  """
  Return all odd primes strictly less than n using the Sieve of
  Eratosthenes.

  Only odd numbers are tracked; 2 is appended by the caller.
  Callers needing primes <= n must pass n+1 (as generate_primes does).

  Parameters
  ----------
  n : int
    Strict upper bound (exclusive).

  Returns
  -------
  primes : NDArray[int64]
    Array of odd primes in [3, n).
  """
  _n = int64(n)
  sieve = ones(_n // 2, dtype=bool_)
  limit = int(sqrt(float64(n))) + 1

  for i in range(3, limit, 2):
    if sieve[i // 2]:
      sieve[i * i // 2 :: i] = False

  prime_indices = nonzero(sieve)[0][1:]
  n_primes = prime_indices.size
  primes = empty(n_primes, dtype=int64)
  for j in range(n_primes):
    primes[j] = 2 * int64(prime_indices[j]) + 1

  return primes


###############################################################################
# Internal: segmented Eratosthenes sieve
###############################################################################
def _seg_sieve_primes(seg_start : int,
                      seg_end : int) -> NDArray[int64]:
  """
  Return primes in the interval [seg_start, seg_end] using a
  segmented Sieve of Eratosthenes.

  Parameters
  ----------
  seg_start : int
    Lower bound (inclusive).
  seg_end : int
    Upper bound (inclusive).

  Returns
  -------
  primes : NDArray[int64]
    Primes in the interval.
  """
  # Snap to odd boundaries
  _s = int64(seg_start)
  _e = int64(seg_end)
  _s = _s if (_s % 2) else _s + 1
  _e = _e if (_e % 2) else _e - 1

  seg_len = _e - _s + 1
  sieve_len = (seg_len + 1) // 2
  sieve = ones(sieve_len, dtype=bool_)

  root_limit = int(sqrt(float64(_e))) + 1
  root_primes = _sieve_primes(root_limit)

  for _rp in root_primes:
    rp = int64(_rp)
    prime_multiple = _s - _s % rp
    while not ((prime_multiple % 2) and (prime_multiple >= _s)):
      prime_multiple += rp

    sieve_start = (prime_multiple - _s) // 2
    sieve[sieve_start : sieve_len : rp] = False

    if _s <= rp <= _e:
      ix = int64((rp - _s) // 2)
      sieve[ix] = True

  prime_indices = nonzero(sieve)[0]
  n_primes = prime_indices.size
  primes = empty(n_primes, dtype=int64)
  for j in range(n_primes):
    primes[j] = 2 * int64(prime_indices[j]) + _s

  return primes


###############################################################################
# Public: generate all primes up to n
###############################################################################
def generate_primes(n : int) -> NDArray[int64]:
  """
  Return all primes less than or equal to n.

  Uses the odds-only Eratosthenes sieve.

  Parameters
  ----------
  n : int
    Upper bound (inclusive).

  Returns
  -------
  primes : NDArray[int64]
    Primes <= n, including 2.

  Usage
  -----
  >>> generate_primes(20)
      array([ 2,  3,  5,  7, 11, 13, 17, 19])
  """
  if n < 2:
    return zeros(0, dtype=int64)
  odd_primes = _sieve_primes(n + 1)
  return concatenate((array([2], dtype=int64), odd_primes))


###############################################################################
# Public: primality test
###############################################################################
def is_prime(n : int) -> bool:
  """
  Test whether n is prime by checking against the sieve output.

  Parameters
  ----------
  n : int
    Integer to test.

  Returns
  -------
  result : bool
    True if n is prime.

  Usage
  -----
  >>> is_prime(17)
      True
  >>> is_prime(18)
      False
  """
  _n = int64(n)
  if _n < 2:
    return False
  if _n == 2:
    return True
  # Trial division up to sqrt(n) to avoid full sieve for large n
  limit = int(sqrt(float(_n)))
  primes = generate_primes(limit)
  for i in range(primes.size):
    p = primes[i]
    if p * p > _n:
      break
    if _n % p == 0:
      return False
  return True


###############################################################################
# Public: prime factorization
###############################################################################
def get_prime_factors(n : int) -> NDArray[int64]:
  """
  Return the prime factorization of n.

  Returns an (N, 2) array where each row is [prime, power].

  Parameters
  ----------
  n : int
    Integer to factorize (>= 2).

  Returns
  -------
  factors : NDArray[int64]
    Array of shape (num_factors, 2).  Column 0: primes, column 1: powers.

  Usage
  -----
  >>> get_prime_factors(84)
      array([[2, 2],
             [3, 1],
             [7, 1]])
  """
  _n = int64(n)
  if _n < 2:
    return zeros((0, 2), dtype=int64)

  pr = generate_primes(n)
  n_pr = pr.size

  n_factors = 0
  _rem = _n
  for idx in range(n_pr):
    p = int64(pr[idx])
    if _rem % p == 0:
      n_factors += 1
      while _rem % p == 0:
        _rem //= p
    if 0 < _rem - 1 < p * p - _rem:
      # p*p > _rem > 1 AND _rem < (p*p+1)/2 (strict chain; not just p*p > _rem)
      n_factors += 1
      break

  # Second pass: fill
  out = empty((n_factors, 2), dtype=int64)
  _rem = _n
  fi = 0
  for idx in range(n_pr):
    p = int64(pr[idx])
    if _rem % p == 0:
      cnt = int64(0)
      while _rem % p == 0:
        cnt += 1
        _rem //= p
      out[fi, 0] = p
      out[fi, 1] = cnt
      fi += 1
    if 0 < _rem - 1 < p * p - _rem:
      # p*p > _rem > 1 AND _rem < (p*p+1)/2 (strict chain; not just p*p > _rem)
      out[fi, 0] = _rem
      out[fi, 1] = int64(1)
      break

  return out[:fi + 1] if n_factors > fi else out[:fi]


###############################################################################
# Internal: greatest common divisor (Euclid, scalar)
###############################################################################
def _gcd_euclid(a : int, b : int) -> int64:
  """
  Euclidean algorithm for gcd of two scalars (iterative).
  """
  _a = int64(a)
  _b = int64(b)
  while _b != 0:
    _a, _b = _b, _a % _b
  return _a


###############################################################################
# Public: greatest common divisor
###############################################################################
def gcd(a : int, b : int) -> int64:
  """
  Compute the greatest common divisor via Euclid's algorithm.

  Parameters
  ----------
  a : int
  b : int

  Returns
  -------
  d : int
    gcd(a, b).

  Usage
  -----
  >>> gcd(48, 18)
      6
  """
  return _gcd_euclid(a, b)


###############################################################################
# Public: least common multiple
###############################################################################
def lcm(a : int, b : int) -> int64:
  """
  Compute the least common multiple: lcm(a, b) = a * b // gcd(a, b).

  Parameters
  ----------
  a : int
  b : int

  Returns
  -------
  m : int64
    lcm(a, b).

  Usage
  -----
  >>> lcm(12, 18)
      36
  """
  return int64(a * b // _gcd_euclid(a, b))


###############################################################################
# Internal: extended Euclidean algorithm (scalar)
###############################################################################
def _lin_diophantine(a : int,
                     b : int) -> NDArray[int64]:
  """
  Extended Euclidean: return (x, y, gcd) such that a*x + b*y = gcd.
  Iterative version for numba compatibility.
  """
  _a = int64(a)
  _b = int64(b)

  x0 = int64(1)
  x1 = int64(0)
  y0 = int64(0)
  y1 = int64(1)

  sign_a = int64(1) if _a >= 0 else int64(-1)
  sign_b = int64(1) if _b >= 0 else int64(-1)
  _a = abs(_a)
  _b = abs(_b)

  while _b != 0:
    q = _a // _b
    _a, _b = _b, _a - q * _b
    x0, x1 = x1, x0 - q * x1
    y0, y1 = y1, y0 - q * y1

  out = empty(3, dtype=int64)
  out[0] = sign_a * x0
  out[1] = sign_b * y0
  out[2] = _a
  return out


###############################################################################
# Public: solve linear Diophantine equation
###############################################################################
def solve_lin_diophantine(a : int,
                          b : int) -> NDArray[int64]:
  """
  Find integer solution (x, y) to a*x + b*y = gcd(a, b).

  Parameters
  ----------
  a : int
  b : int

  Returns
  -------
  result : NDArray[int64]
    Array [x, y, gcd] such that a*x + b*y = gcd.

  Usage
  -----
  >>> solve_lin_diophantine(48, 18)
      array([-1,  3,  6])
  >>> # Verify: 48*(-1) + 18*3 = -48 + 54 = 6 = gcd(48, 18)
  """
  return _lin_diophantine(a, b)


###############################################################################
# Public: next power of two
###############################################################################
def next_pow2(N : int) -> int64:
  """
  Find the smallest p such that 2**p >= 2*N + 1.

  Parameters
  ----------
  N : int
    Input integer.

  Returns
  -------
  p : int
    Exponent.

  Usage
  -----
  >>> next_pow2(8)
      5
  >>> # 2**5 = 32 >= 2*8+1 = 17
  """
  _N = int64(N)
  p = int64(1)
  while (int64(1) << p) < 2 * _N + 1:
    p += 1
  return p


###############################################################################
# Public: number of divisors
###############################################################################
def get_num_divisors(n : int) -> int64:
  r"""
  Return the number of positive divisors of n.

  Uses prime factorization: :math:`d(n) = \prod (e_i + 1)` for
  :math:`n = \prod p_i^{e_i}`.

  Parameters
  ----------
  n : int
    Input integer (>= 1).

  Returns
  -------
  nd : int
    Number of divisors.

  Usage
  -----
  >>> get_num_divisors(12)
      6
  """
  if n == 1:
    return int64(1)
  factors = get_prime_factors(n)
  if factors.size == 0:
    return int64(1)
  nd = int64(1)
  for i in range(factors.shape[0]):
    nd *= (int64(factors[i, 1]) + 1)
  return nd


###############################################################################
# Internal: prime sqrt decomposition
###############################################################################
def _prime_sqrt_decomp(i_sqrt_N : int,
                       P_arr : NDArray[int64],
                       A_arr : NDArray[int64]) -> int64:
  """
  Find the largest i <= i_sqrt_N such that every prime factor of i
  appears in P_arr with power <= the corresponding entry in A_arr.

  P_arr and A_arr are assumed sorted by prime.
  """
  _i = int64(i_sqrt_N)
  n_factors = P_arr.size

  while _i >= 1:
    _rem = _i
    ok = True
    for j in range(n_factors):
      p = int64(P_arr[j])
      a_max = int64(A_arr[j])
      cnt = int64(0)
      while _rem % p == 0:
        cnt += 1
        _rem //= p
      if cnt > a_max:
        ok = False
        break
    # Check if remaining factor is a prime not in P_arr
    if ok and _rem > 1:
      found = False
      for j in range(n_factors):
        if int64(P_arr[j]) == _rem:
          found = True
          break
      if not found:
        ok = False

    if ok:
      return _i
    _i -= 1

  return int64(1)


###############################################################################
# Public: balanced prime decomposition for Cooley-Tukey splits
###############################################################################
def prime_sqrt_decomp(N : int) -> int64:
  """
  Given N, find the largest integer i <= sqrt(N) whose prime
  factors are a subset of the prime factors of N (with powers
  not exceeding those in N's factorization).

  Useful for choosing balanced factors in Cooley-Tukey FFT splits.

  Parameters
  ----------
  N : int
    Integer to decompose.

  Returns
  -------
  i : int
    Factor i such that 1 <= i <= sqrt(N) and i's prime
    factorization is compatible with N's.

  Usage
  -----
  >>> prime_sqrt_decomp(60)
      6
  """
  _N = int64(N)
  if _N <= 1:
    return int64(1)

  factors = get_prime_factors(N)
  if factors.size == 0:
    return int64(1)

  P_arr = factors[:, 0].copy()
  A_arr = factors[:, 1].copy()

  i_sqrt = int(sqrt(float64(_N)))
  return _prime_sqrt_decomp(i_sqrt, P_arr, A_arr)


###############################################################################
# Public: expand (prime, power) pairs to flat repeated-factor array
###############################################################################
def prime_power_arrs_to_list(primes : NDArray[int64],
                             powers : NDArray[int64]) -> NDArray[int64]:
  """
  Expand arrays of (prime, exponent) into a flat array of repeated
  prime factors.

  Parameters
  ----------
  primes : NDArray[int64]
    Prime factors.
  powers : NDArray[int64]
    Corresponding exponents.

  Returns
  -------
  result : NDArray[int64]
    Flat array where each prime appears exponent times.

  Usage
  -----
  >>> prime_power_arrs_to_list(array([2, 3, 7]), array([2, 1, 1]))
      array([2, 2, 3, 7])
  """
  n_pairs = primes.size
  total = int64(0)
  for i in range(n_pairs):
    total += int64(powers[i])

  out = empty(total, dtype=int64)
  pos = 0
  for i in range(n_pairs):
    p = int64(primes[i])
    cnt = int64(powers[i])
    for _ in range(cnt):
      out[pos] = p
      pos += 1

  return out


###############################################################################
# JIT if not type-checking as required
###############################################################################
if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  _sieve_primes = JIT(_sieve_primes)
  _seg_sieve_primes = JIT(_seg_sieve_primes)
  generate_primes = JIT(generate_primes)
  is_prime = JIT(is_prime)
  get_prime_factors = JIT(get_prime_factors)
  _gcd_euclid = JIT(_gcd_euclid)
  gcd = JIT(gcd)
  lcm = JIT(lcm)
  _lin_diophantine = JIT(_lin_diophantine)
  solve_lin_diophantine = JIT(solve_lin_diophantine)
  _prime_sqrt_decomp = JIT(_prime_sqrt_decomp)
  prime_sqrt_decomp = JIT(prime_sqrt_decomp)
  get_num_divisors = JIT(get_num_divisors)
  next_pow2 = JIT(next_pow2)
  prime_power_arrs_to_list = JIT(prime_power_arrs_to_list)

#
# :D
#
