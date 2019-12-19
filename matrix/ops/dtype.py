import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import Dense
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Dense)
def dtype(a):
    return B.dtype(a.mat)


@B.dispatch(Diagonal)
def dtype(a):
    return B.dtype(a.diag)


@B.dispatch(Constant)
def dtype(a):
    return B.dtype(a.const)


@B.dispatch(LowRank)
def dtype(a):
    return B.dtype(a.left)


@B.dispatch(Woodbury)
def dtype(a):
    return B.dtype(a.lr)
