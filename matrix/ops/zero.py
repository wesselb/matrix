import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def zero(a):
    return B.zero(B.dtype(a))
