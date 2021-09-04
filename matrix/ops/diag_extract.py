import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch
def diag_extract(a: AbstractMatrix):
    return B.diag(a)
