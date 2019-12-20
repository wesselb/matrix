import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import Dense
from ..woodbury import Woodbury
from ..kronecker import Kronecker

__all__ = []


def _diag_len(a):
    return B.minimum(*B.shape(a))


@B.dispatch(Zero)
def diag(a):
    return B.zeros(B.dtype(a), _diag_len(a))


@B.dispatch(Dense)
def diag(a):
    return B.diag(a.mat)


@B.dispatch(Diagonal)
def diag(a):
    return a.diag


@B.dispatch(Constant)
def diag(a):
    return a.const * B.ones(B.dtype(a), _diag_len(a))


@B.dispatch(LowRank)
def diag(a):
    diag_len = _diag_len(a)
    return B.sum(a.left[:diag_len, :] * a.right[:diag_len, :], axis=1)


@B.dispatch(Woodbury)
def diag(a):
    return B.diag(a.diag) + B.diag(a.lr)


@B.dispatch(Kronecker)
def diag(a):
    return B.kron(B.diag(a.left), B.diag(a.right))
