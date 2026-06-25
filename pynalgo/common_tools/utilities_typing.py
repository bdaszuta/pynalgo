"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Type hint re-exports: Any, TYPE_CHECKING, Optional, NDArray, float64, complex128, int64.
"""
__all__ = ["Any", "TYPE_CHECKING", "Optional", "NDArray",
           "float64", "complex128", "int64"]

from typing import Any, TYPE_CHECKING, Optional
from numpy.typing import (NDArray, )
from numpy import (float64, complex128, int64)

#
# :D
#
