import lab as B

from ..matrix import AbstractMatrix
from ..shape import assert_square

__all__ = []


@B.dispatch(AbstractMatrix)
def eye(a):
    assert_square(a, "Can only construct identity matrices from square matrices.")
    return B.fill_diag(B.one(a), B.shape(a)[0])
