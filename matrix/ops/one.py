import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def one(a: AbstractMatrix):
    return B.one(B.dtype(a))