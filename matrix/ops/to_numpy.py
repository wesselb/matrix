import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def to_numpy(a):
    return B.to_numpy(B.dense(a))
