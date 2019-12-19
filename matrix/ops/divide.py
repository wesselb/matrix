import lab as B

from ..matrix import Dense

__all__ = []


@B.dispatch(Dense, Dense)
def divide(a, b):
    return Dense(B.divide(a.mat, b.mat))
