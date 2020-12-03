import lab as B

from ..diagonal import Diagonal
from ..matrix import Dense
from ..constant import Zero, Constant
from ..triangular import LowerTriangular, UpperTriangular
from ..lowrank import LowRank
from ..woodbury import Woodbury
from ..kronecker import Kronecker

__all__ = []


@B.dispatch(Zero)
def negative(a):
    return a


@B.dispatch(Dense)
def negative(a):
    return Dense(B.negative(a.mat))


@B.dispatch(Diagonal)
def negative(a):
    return Diagonal(B.negative(a.diag))


@B.dispatch(Constant)
def negative(a):
    return Constant(-a.const, a.rows, a.cols)


@B.dispatch(LowerTriangular)
def negative(a):
    return LowerTriangular(B.negative(a.mat))


@B.dispatch(UpperTriangular)
def negative(a):
    return UpperTriangular(B.negative(a.mat))


@B.dispatch(LowRank)
def negative(a):
    return LowRank(a.left, a.right, B.negative(a.middle))


@B.dispatch(Woodbury)
def negative(a):
    return Woodbury(B.negative(a.diag), B.negative(a.lr))


@B.dispatch(Kronecker)
def negative(a):
    return Kronecker(B.negative(a.left), a.right)
