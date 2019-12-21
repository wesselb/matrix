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
def add(a, b):
    assert_compatible(a, b)
    return a


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def add(a, b):
    assert_compatible(a, b)
    return b


@B.dispatch(Dense, Dense)
def add(a, b):
    return Dense(B.add(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def add(a, b):
    return Diagonal(B.add(a.diag, b.diag))


@B.dispatch(Constant, Constant)
def add(a, b):
    assert_compatible(a, b)
    return Constant(a.const + b.const, *broadcast(a, b).as_tuple())


@B.dispatch(Constant, AbstractMatrix)
def add(a, b):
    assert_compatible(a, b)
    return Dense(a.const + B.dense(b))


@B.dispatch(AbstractMatrix, Constant)
def add(a, b):
    return add(b, a)


@B.dispatch(LowRank, LowRank)
def add(a, b):
    assert_compatible(a, b)
    return LowRank(B.concat(a.left, b.left, axis=1),
                   B.concat(a.right, b.right, axis=1))


@B.dispatch(Constant, LowRank)
def add(a, b):
    assert_compatible(a, b)
    dtype = B.dtype(b)
    rows, cols = B.shape(b)
    ones_left = B.ones(dtype, rows, 1)
    ones_right = B.ones(dtype, cols, 1)
    return LowRank(B.concat(ones_left, b.left, axis=1),
                   B.concat(a.const * ones_right, b.right, axis=1))


@B.dispatch(LowRank, Constant)
def add(a, b):
    return add(b, a)


@B.dispatch(Diagonal, LowRank)
def add(a, b):
    return Woodbury(a, b)


@B.dispatch(LowRank, Diagonal)
def add(a, b):
    return Woodbury(b, a)


@B.dispatch(Woodbury, Woodbury)
def add(a, b):
    return Woodbury(a.diag + b.diag, a.lr + b.lr)


@B.dispatch(Woodbury, Diagonal)
def add(a, b):
    return Woodbury(a.diag + b, a.lr)


@B.dispatch(Diagonal, Woodbury)
def add(a, b):
    return add(b, a)


@B.dispatch.multi((Woodbury, Constant),
                  (Woodbury, LowRank))
def add(a, b):
    return Woodbury(a.diag, a.lr + b)


@B.dispatch.multi((Constant, Woodbury),
                  (LowRank, Woodbury))
def add(a, b):
    return add(b, a)
