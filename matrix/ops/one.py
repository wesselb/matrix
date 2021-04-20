import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def one(a):
    return B.ones(B.dtype(a), B.shape_batch(a))
