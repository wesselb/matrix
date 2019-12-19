import lab as B

from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense
from ..constant import Constant
from ..shape import assert_compatible, broadcast

__all__ = []


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
