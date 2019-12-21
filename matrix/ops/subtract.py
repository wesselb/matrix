import lab as B
from algebra import proven

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense
from ..shape import assert_compatible, broadcast
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def subtract(a, b):
    assert_compatible(a, b)
    return a


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def subtract(a, b):
    assert_compatible(a, b)
    return B.negative(b)


@B.dispatch(Dense, Dense)
def subtract(a, b):
    return Dense(B.subtract(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def subtract(a, b):
    return Diagonal(B.subtract(a.diag, b.diag))


@B.dispatch(Constant, Constant)
def subtract(a, b):
    assert_compatible(a, b)
    return Constant(B.subtract(a.const, b.const), *broadcast(a, b).as_tuple())


@B.dispatch(Constant, AbstractMatrix)
def subtract(a, b):
    assert_compatible(a, b)
    return Dense(B.subtract(a.const, B.dense(b)))


@B.dispatch(AbstractMatrix, Constant)
def subtract(a, b):
    assert_compatible(a, b)
    return Dense(B.subtract(B.dense(a), b.const))


@B.dispatch.multi((LowRank, LowRank),
                  (Constant, LowRank),
                  (LowRank, Constant),
                  (Woodbury, Woodbury),
                  (Woodbury, Diagonal),
                  (Diagonal, Woodbury),
                  (Woodbury, Constant),
                  (Woodbury, LowRank),
                  (Constant, Woodbury),
                  (LowRank, Woodbury))
def subtract(a, b):
    return B.add(a, B.negative(b))
