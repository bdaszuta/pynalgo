"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: JIT decorator variants (JIT, JITI, JITG, JIT_NC, JITI_NC)
           and idx_interchange.
"""
from functools import partial
import inspect

from numpy import (asarray, empty)
from numba import jit, njit
from numba.extending import overload

from pynalgo.common_tools.utilities_typing import (NDArray, Any, int64,
                                                   TYPE_CHECKING)


def generated_jit(function=None, **options):
    """Drop-in replacement for numba.generated_jit (removed >= 0.60).

    Uses @overload for compile-time type specialization with an
    @njit wrapper providing the Python-callable entry point.
    Per-ndim specialization preserved — callable from both Python
    and nopython contexts.

    The wrapper uses explicit parameter names (not *args) so that
    Python fills in default values before entering njit, ensuring
    the overload always sees explicit argument types.

    Parameters
    ----------
    function : callable or None
        Type-inspector function receiving numba types and returning
        a specialized callable implementation.
    **options
        JIT options forwarded to @njit and @overload.
    """
    options.pop('nopython', None)          # deprecated kwarg for njit

    jit_options = {k: v for k, v in options.items()
                   if k != 'error_model'}   # overload doesn't accept this

    def _decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        defaults = {n: p.default for n, p in sig.parameters.items()
                    if p.default is not inspect.Parameter.empty}

        # Build stub with explicit parameter names
        stub_src = ('def _stub(' + ', '.join(param_names) +
                    '):\n    raise NotImplementedError')
        ns = {}
        exec(stub_src, ns)
        _stub = ns['_stub']

        overload(_stub, strict=False, jit_options=jit_options)(func)

        # Build wrapper with explicit params + Python-level defaults
        params_with_defaults = []
        for name in param_names:
            if name in defaults:
                params_with_defaults.append(f'{name}={defaults[name]!r}')
            else:
                params_with_defaults.append(name)
        wrapper_src = (
            'def _jit_wrapper(' + ', '.join(params_with_defaults) + '):\n'
            '    return _stub(' + ', '.join(param_names) + ')'
        )
        exec(wrapper_src, ns)
        _jit_wrapper = ns['_jit_wrapper']

        # Cache is disabled for the thin njit wrapper (exec'd code
        # has no source file).  The @overload mechanism caches each
        # per-type specialization independently.
        options['cache'] = False
        return njit(**options)(_jit_wrapper)

    if function is None:
        return _decorator
    return _decorator(function)


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
    Perform operation in-place (requires ndarray input;
    other array-like types are copied).

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
