from typing import Union

import lab as B

from ..matrix import AbstractMatrix
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


@B.dispatch
def cholesky_solve(a: Union[LowerTriangular, UpperTriangular], b: AbstractMatrix):
    return B.solve(B.transpose(a), B.solve(a, b))
