import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def subtract(a: AbstractMatrix, b: AbstractMatrix):
    return B.add(a, B.negative(b))
