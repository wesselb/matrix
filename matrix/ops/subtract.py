import lab as B
from algebra import proven

from ..matrix import AbstractMatrix

__all__ = []


# Zero


@B.dispatch(precedence=proven())
def subtract(a: AbstractMatrix, b: AbstractMatrix):
    return B.add(a, B.negative(b))