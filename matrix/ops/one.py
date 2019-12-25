import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def one(a):
    return B.one(B.dtype(a))
