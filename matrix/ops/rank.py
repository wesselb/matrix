import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def rank(a: AbstractMatrix):
    return len(B.shape(a))
