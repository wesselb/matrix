import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense
from ..shape import assert_compatible, broadcast

__all__ = []


@B.dispatch(Dense, Dense)
def add(a, b):
    return Dense(B.add(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def add(a, b):
    return Diagonal(B.add(a.diag, b.diag))


@B.dispatch(Constant, Constant)
def add(a, b):
    assert_compatible(a, b)
    return Constant(a.const + b.const, *broadcast(a, b))


@B.dispatch(Constant, AbstractMatrix)
def add(a, b):
    assert_compatible(a, b)
    return Dense(a.const + B.dense(b))


@B.dispatch(AbstractMatrix, Constant)
def add(a, b):
    assert_compatible(a, b)
    return Dense(B.dense(a) + b.const)
