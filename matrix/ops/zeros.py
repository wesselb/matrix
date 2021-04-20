import lab as B

from ..matrix import AbstractMatrix
from ..constant import Zero

__all__ = []


@B.dispatch
def zeros(a: AbstractMatrix):
    return Zero(B.dtype(a), *B.shape(a))