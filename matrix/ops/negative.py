import lab as B

from ..diagonal import Diagonal
from ..matrix import Dense
from ..constant import Constant
from ..lowrank import LowRank
from ..woodbury import Woodbury
from ..kronecker import Kronecker

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


@B.dispatch(LowRank)
def negative(a):
    return LowRank(-a.left, a.right)


@B.dispatch(Woodbury)
def negative(a):
    return Woodbury(B.negative(a.diag), B.negative(a.lr))


@B.dispatch(Kronecker)
def negative(a):
    return Kronecker(B.negative(a.left), a.right)
