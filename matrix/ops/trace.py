import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def trace(a):
    # The implementation of diagonal is optimised, so this should be efficient.
    return B.sum(B.diag(a))
