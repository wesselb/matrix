import lab as B
from algebra import proven

from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense
from ..constant import Zero, Constant
from ..lowrank import LowRank
from ..woodbury import Woodbury
from ..shape import assert_compatible, broadcast

__all__ = []


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def multiply(a, b):
    assert_compatible(a, b)
    return b


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def multiply(a, b):
    assert_compatible(a, b)
    return a


@B.dispatch(Dense, Dense)
def multiply(a, b):
    return Dense(B.multiply(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def multiply(a, b):
    return Diagonal(B.multiply(a.diag, b.diag))


@B.dispatch(Constant, Constant)
def multiply(a, b):
    assert_compatible(a, b)
    return Constant(a.const * b.const, *broadcast(a, b))


@B.dispatch(Constant, AbstractMatrix)
def multiply(a, b):
    assert_compatible(a, b)
    return Dense(a.const * B.dense(b))


@B.dispatch(AbstractMatrix, Constant)
def multiply(a, b):
    assert_compatible(a, b)
    return Dense(B.dense(a) * b.const)


@B.dispatch(Constant, Diagonal)
def multiply(a, b):
    assert_compatible(a, b)
    return Diagonal(a.const * b.diag)


@B.dispatch(Diagonal, Constant)
def multiply(a, b):
    assert_compatible(a, b)
    return Diagonal(b.const * a.diag)


@B.dispatch(LowRank, LowRank)
def multiply(a, b):
    assert_compatible(a, b)

    # Pick apart the matrices.
    al, ar = B.unstack(a.left, axis=1), B.unstack(a.right, axis=1)
    bl, br = B.unstack(b.left, axis=1), B.unstack(b.right, axis=1)

    # Construct the factors.
    left = B.stack(*[ali * blk
                     for ali in al
                     for blk in bl], axis=1)
    right = B.stack(*[arj * brl
                      for arj in ar
                      for brl in br], axis=1)

    return LowRank(left, right)
