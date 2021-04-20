import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def shape(a):
    return B.shape_batch(a) + B.shape_matrix(a)
