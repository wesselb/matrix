import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def uprank(a: AbstractMatrix, rank: B.Int = 2):
    if rank != 2:
        raise ValueError("Can only uprank a matrix to rank two.")
    return a
