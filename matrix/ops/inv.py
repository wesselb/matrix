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
    # Explicitly computing the inverse is not great numerically, but solving
    # against left or right destroys symmetry, which hinders further algebraic
    # simplifications.
    return B.subtract(diag_inv, LowRank(
        B.matmul(diag_inv, a.lr.left),
        B.matmul(diag_inv, a.lr.right),
        B.inv(B.schur(a))
    ))


@B.dispatch(Kronecker)
def inv(a):
    return Kronecker(B.inv(a.left), B.inv(a.right))
