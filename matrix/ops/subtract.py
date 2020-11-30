import lab as B
from algebra import proven

from ..matrix import AbstractMatrix

__all__ = []


# Zero


@B.dispatch(AbstractMatrix, AbstractMatrix, precedence=proven())
def subtract(a, b):
    return B.add(a, B.negative(b))
