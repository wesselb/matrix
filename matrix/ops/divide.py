import lab as B
from plum import promote

from ..matrix import AbstractMatrix, Dense

__all__ = []


@B.dispatch({B.Numeric, AbstractMatrix}, {B.Numeric, AbstractMatrix})
def divide(a, b):
    return B.divide(*promote(a, b))


@B.dispatch(Dense, Dense)
def divide(a, b):
    return Dense(B.divide(a.mat, b.mat))
