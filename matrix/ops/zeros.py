import lab as B

from ..constant import Zero
from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def zeros(a: AbstractMatrix):
    return Zero(B.dtype(a), *B.shape(a))
