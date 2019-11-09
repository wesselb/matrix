import lab as B
from plum import promote

from ..matrix import AbstractMatrix, Dense
from ..diagonal import Diagonal

__all__ = []


@B.dispatch({B.Numeric, AbstractMatrix}, {B.Numeric, AbstractMatrix})
def add(a, b):
    return B.add(*promote(a, b))


@B.dispatch(Dense, Dense)
def add(a, b):
    return Dense(B.add(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def add(a, b):
    return Diagonal(B.add(a.diag, b.diag))
