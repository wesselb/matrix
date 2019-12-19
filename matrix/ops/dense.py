import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..matrix import Dense
from ..lowrank import LowRank
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(B.Numeric)
def dense(a):
    return a


B.dense = dense


@B.dispatch(Dense)
def dense(a):
    return a.mat


@B.dispatch(Diagonal)
def dense(a):
    return B.diag(a.diag)


@B.dispatch(Constant)
def dense(a):
    return a.const * B.ones(B.dtype(a.const), a.rows, a.cols)


@B.dispatch(LowRank)
def dense(a):
    return B.outer(a.left, a.right)


@B.dispatch(Woodbury)
def dense(a):
    return B.dense(a.diag) + B.dense(a.lr)
