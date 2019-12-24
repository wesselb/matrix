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
    if a.dense is None:
        a.dense = B.zeros(a.dtype, a.rows, a.cols)
    return a.dense


@B.dispatch(Dense)
def dense(a):
    return a.mat


@B.dispatch(Diagonal)
def dense(a):
    if a.dense is None:
        a.dense = B.diag(a.diag)
    return a.dense


@B.dispatch(Constant)
def dense(a):
    if a.dense is None:
        a.dense = a.const * B.ones(B.dtype(a.const), a.rows, a.cols)
    return a.dense


@B.dispatch(LowRank)
def dense(a):
    if a.dense is None:
        a.dense = B.dense(B.matmul(a.left, a.right, tr_b=True))
    return a.dense


@B.dispatch(Woodbury)
def dense(a):
    if a.dense is None:
        a.dense = B.dense(a.diag) + B.dense(a.lr)
    return a.dense


@B.dispatch(Kronecker)
def dense(a):
    if a.dense is None:
        a.dense = B.kron(B.dense(a.left), B.dense(a.right))
    return a.dense
