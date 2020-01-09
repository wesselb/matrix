import lab as B

from ..matrix import AbstractMatrix
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


@B.dispatch({LowerTriangular, UpperTriangular}, AbstractMatrix)
def cholesky_solve(a, b):
    return B.solve(B.transpose(a), B.solve(a, b))
