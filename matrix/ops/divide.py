import lab as B

from ..matrix import AbstractMatrix, Dense

__all__ = []


@B.dispatch(AbstractMatrix, AbstractMatrix)
def divide(a, b):
    return Dense(B.divide(B.dense(a), B.dense(b)))
