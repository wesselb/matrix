import lab as B

from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import Dense
from ..lowrank import LowRank
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(Dense)
def inv(a):
    return Dense(B.inv(a.mat))


@B.dispatch(Diagonal)
def inv(a):
    return Diagonal(B.divide(1, a.diag))


@B.dispatch(Woodbury)
def inv(a):
    diag_inv = B.inv(a.diag)
    return B.subtract(diag_inv,
                      LowRank(B.matmul(diag_inv, a.lr.left),
                              B.matmul(diag_inv, a.lr.right),
                              B.inv(B.dense(B.schur(a)))))


@B.dispatch(Kronecker)
def inv(a):
    return Kronecker(B.inv(a.left), B.inv(a.right))
