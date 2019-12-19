import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..matrix import Dense

__all__ = []


@B.dispatch(Zero, B.Numeric)
def power(a, b):
    return a


@B.dispatch(Dense, B.Numeric)
def power(a, b):
    return Dense(B.power(a.mat, b))


@B.dispatch(Diagonal, B.Numeric)
def power(a, b):
    return Diagonal(B.power(a.diag, b))


@B.dispatch(Constant, B.Numeric)
def power(a, b):
    return Constant(B.power(a.const, b), a.rows, a.cols)
