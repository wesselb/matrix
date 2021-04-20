import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def zero(a: AbstractMatrix):
    return B.zero(B.dtype(a))