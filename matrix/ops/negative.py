import lab as B

from ..diagonal import Diagonal
from ..matrix import Dense
from ..constant import Constant

__all__ = []


@B.dispatch(Dense)
def negative(a):
    return Dense(B.negative(a.mat))


@B.dispatch(Diagonal)
def negative(a):
    return Diagonal(B.negative(a.diag))


@B.dispatch(Constant)
def negative(a):
    return Constant(-a.const, a.rows, a.cols)
