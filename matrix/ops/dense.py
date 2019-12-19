import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..matrix import Dense

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
