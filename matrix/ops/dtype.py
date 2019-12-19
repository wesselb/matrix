import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..matrix import Dense

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
