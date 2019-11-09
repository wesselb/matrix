import lab as B
from ..matrix import Dense
from ..diagonal import Diagonal

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
