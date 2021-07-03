import lab as B

from ..constant import Constant
from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def ones(a: AbstractMatrix):
    return Constant(B.one(a), *B.shape(a))
