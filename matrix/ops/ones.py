import lab as B

from ..matrix import AbstractMatrix
from ..constant import Constant

__all__ = []


@B.dispatch(AbstractMatrix)
def ones(a):
    return Constant(B.one(a), *B.shape(a))
