import lab as B

from ..diagonal import Diagonal
from ..matrix import Dense
from ..constant import Constant
from ..lowrank import LowRank
from ..woodbury import Woodbury
from ..kronecker import Kronecker

__all__ = []


@B.dispatch(Dense)
def transpose(a):
    return Dense(B.transpose(a.mat))


@B.dispatch(Diagonal)
def transpose(a):
    return a


@B.dispatch(Constant)
def transpose(a):
    return Constant(a.const, a.cols, a.rows)


@B.dispatch(LowRank)
def transpose(a):
    return LowRank(a.right, a.left)


@B.dispatch(Woodbury)
def transpose(a):
    return Woodbury(B.transpose(a.diag), B.transpose(a.lr))


@B.dispatch(Kronecker)
def transpose(a):
    return Kronecker(B.transpose(a.left), B.transpose(a.right))
