import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def uprank(a: AbstractMatrix):
    return a