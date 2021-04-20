import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def to_numpy(a: AbstractMatrix):
    return B.to_numpy(B.dense(a))