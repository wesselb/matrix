import lab as B

from ..diagonal import Diagonal
from ..matrix import Dense

__all__ = []


@B.dispatch(Dense)
def negative(a):
    return Dense(B.negative(a.mat))


@B.dispatch(Diagonal)
def negative(a):
    return Diagonal(B.negative(a.diag))
