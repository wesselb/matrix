import lab as B

from ..matrix import AbstractMatrix
from ..constant import Zero

__all__ = []


@B.dispatch(AbstractMatrix)
def zeros(a):
    return Zero(B.dtype(a), *B.shape(a))
