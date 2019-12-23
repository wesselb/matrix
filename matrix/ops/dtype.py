import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Zero)
def dtype(a):
    return a.dtype


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
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


@B.dispatch(Kronecker)
def dtype(a):
    return B.dtype(a.left)
