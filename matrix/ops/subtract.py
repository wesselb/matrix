import lab as B
from plum import promote

from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense

__all__ = []


@B.dispatch({B.Numeric, AbstractMatrix}, {B.Numeric, AbstractMatrix})
def subtract(a, b):
    return B.subtract(*promote(a, b))


@B.dispatch(Dense, Dense)
def subtract(a, b):
    return Dense(B.subtract(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def subtract(a, b):
    return Diagonal(B.subtract(a.diag, b.diag))
