import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def uprank(a):
    return a
