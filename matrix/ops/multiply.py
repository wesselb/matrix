import lab as B
from plum import promote

from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense

__all__ = []


@B.dispatch({B.Numeric, AbstractMatrix}, {B.Numeric, AbstractMatrix})
def multiply(a, b):
    return B.multiply(*promote(a, b))


@B.dispatch(Dense, Dense)
def multiply(a, b):
    return Dense(B.multiply(a.mat, b.mat))


@B.dispatch(Diagonal, Diagonal)
def multiply(a, b):
    return Diagonal(B.multiply(a.diag, b.diag))
