from functools import reduce
from operator import mul

import lab as B
from lab.util import resolve_axis
from plum import Union

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def mean(a: AbstractMatrix, axis: Union[B.Int, None] = None):
    if axis is None:
        n = reduce(mul, B.shape(a), 1)
    else:
        n = B.shape(a, resolve_axis(a, axis))
    return B.sum(a, axis=axis) / n
