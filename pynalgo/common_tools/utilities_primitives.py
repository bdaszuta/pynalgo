"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Primitive predicates: is_even, is_odd.
"""
from pynalgo.common_tools.utilities_numba import (JITI, TYPE_CHECKING)

def is_odd(val : int) -> bool:
  """
  Is an integer odd?

  Parameters
  ----------
  val : int
    Input integer to test for oddness.

  Returns
  -------
  result : bool
    True if odd.

  Usage
  -----
  >>> is_odd(4)
      False
  >>> is_odd(5)
      True
  """
  return (val % 2) != 0

def is_even(val : int) -> bool:
  """
  Is an integer even?

  Parameters
  ----------
  val : int
    Input integer to test for evenness.

  Returns
  -------
  result : bool
    True if even.

  Usage
  -----
  >>> is_even(4)
      True
  >>> is_even(5)
      False
  """
  return (val % 2) == 0

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  is_odd = JITI(is_odd)
  is_even = JITI(is_even)

#
# :D
#
