import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..matrix import Dense
from ..lowrank import LowRank
from ..woodbury import Woodbury
from ..kronecker import Kronecker

__all__ = []


@B.dispatch(B.Numeric)
def dense(a):
    return a


B.dense = dense


@B.dispatch(Zero)
def dense(a):
    return B.zeros(a.dtype, a.rows, a.cols)


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
    return B.dense(B.matmul(a.left, a.right, tr_b=True))


@B.dispatch(Woodbury)
def dense(a):
    return B.dense(a.diag) + B.dense(a.lr)


@B.dispatch(Kronecker)
def dense(a):
    return B.kron(B.dense(a.left), B.dense(a.right))
