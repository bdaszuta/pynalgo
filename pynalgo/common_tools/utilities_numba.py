"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: JIT decorator variants (JIT, JITI, JITG, JIT_NC, JITI_NC)
           and idx_interchange.
"""
from functools import partial

from numpy import (asarray, empty)
try:
    from numba import jit, generated_jit
except ImportError:
    from numba import jit

    # `generated_jit` removed in numba >= 0.60.
    # Provide a wrapper that forwards to jit so that imports
    # succeed for documentation builds.
    def generated_jit(function=None, **options):
        if function is None:
            return lambda f: jit(f, **options)
        return jit(function, **options)

from pynalgo.common_tools.utilities_typing import (NDArray, Any, int64,
                                                   TYPE_CHECKING)

###############################################################################
# numba settings and convenience function
###############################################################################

# keyword arguments that are fed to each JIT decorator.
SETTINGS_JITI = {'nopython': True,
                 'nogil': True,
                 'cache': True,
                 'inline': 'always',
                 'error_model': 'numpy',
                 'boundscheck': False}

SETTINGS_JIT = {'nopython': True,
                'nogil': True,
                'cache': True,
                'inline': 'never',
                'error_model': 'numpy',
                'boundscheck': False}

SETTINGS_JITI_NC = {'nopython': True,
                    'nogil': True,
                    'cache': False,
                    'inline': 'always',
                    'error_model': 'numpy',
                    'boundscheck': False}

SETTINGS_JIT_NC = {'nopython': True,
                   'nogil': True,
                   'cache': False,
                   'inline': 'never',
                   'error_model': 'numpy',
                   'boundscheck': False}

JITI = partial(jit, **SETTINGS_JITI)
JITI_NC = partial(jit, **SETTINGS_JITI_NC)
JIT = partial(jit, **SETTINGS_JIT)
JIT_NC = partial(jit, **SETTINGS_JIT_NC)
JITG = partial(generated_jit, **SETTINGS_JIT)

###############################################################################
# iterable / array_like manipulations
###############################################################################

def idx_interchange(
  iterable : Any, idx : int, jdx : int, inplace : bool=False
) -> NDArray[int64]:
  '''
  Interchange elements at two input idx locations via a copy.

  Parameters
  ----------
  iterable : Any
    Tuple or array to interchange elements of.

  idx : int
    Index of an element to interchange.

  jdx : int
    Index of an element to interchange.

  inplace : bool = False
    Perform operation in-place (requires ndarray input; other array-like types are copied).

  Returns
  -------
  arr_it : NDArray[int64]
    Resulting interchange as array.

  Usage
  -----
  >>> from numpy import array
  >>> # copy
  >>> idx = (0, 1, 2)
  >>> jdx = idx_interchange(idx, 1, 2)
  >>> array(idx)
      array([0, 1, 2])
  >>> array(jdx)
      array([0, 2, 1])

  >>> from numpy import array
  >>> # in-place
  >>> idx = array([0, 1, 2])
  >>> idx_interchange(idx, 1, 2, inplace=True)
      array([0, 2, 1])
  >>> idx
      array([0, 2, 1])
  '''
  if inplace:
    iterable = asarray(iterable)
    iterable[idx], iterable[jdx] = iterable[jdx], iterable[idx]
    return iterable

  arr_it = empty(len(iterable), int64)
  for i in range(len(iterable)):
    arr_it[i] = iterable[i]

  arr_it[idx], arr_it[jdx] = iterable[jdx], iterable[idx]
  return arr_it

###############################################################################
# JIT if not type-checking as required
###############################################################################

if TYPE_CHECKING:
  reveal_locals()  # noqa: F821
else:
  idx_interchange = JITI(idx_interchange)


#
# :D
#
