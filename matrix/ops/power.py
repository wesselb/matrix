import lab as B

from ..matrix import Dense
from ..diagonal import Diagonal

__all__ = []


@B.dispatch(Dense, object)
def power(a, b):
    return Dense(B.power(a.mat, b))


@B.dispatch(Diagonal, object)
def power(a, b):
    return Diagonal(B.power(a.diag, b))
