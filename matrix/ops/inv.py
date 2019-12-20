import lab as B

from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import Dense
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
    pass  # TODO: Use MIL here.


@B.dispatch(Kronecker)
def inv(a):
    return Kronecker(B.inv(a.left), B.inv(a.right))
