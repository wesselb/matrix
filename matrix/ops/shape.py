import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import Dense
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Dense)
def shape(a):
    return B.shape(a.mat)


@B.dispatch(Diagonal)
def shape(a):
    diag_len = B.length(a.diag)
    return diag_len, diag_len


@B.dispatch(Constant)
def shape(a):
    return a.rows, a.cols


@B.dispatch(LowRank)
def shape(a):
    return B.shape(a.left)[0], B.shape(a.right)[0]


@B.dispatch(Woodbury)
def shape(a):
    return B.shape(a.lr)
