import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def diag_extract(a):
    return B.diag(a)
