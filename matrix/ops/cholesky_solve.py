from typing import Union

import lab as B

from ..matrix import AbstractMatrix
from ..kronecker import Kronecker
from ..diagonal import Diagonal
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


@B.dispatch
def cholesky_solve(
    a: Union[Diagonal, LowerTriangular, UpperTriangular],
    b: AbstractMatrix,
):
    return B.solve(B.transpose(a), B.solve(a, b))


@B.dispatch
def cholesky_solve(a: Kronecker, b: Kronecker):
    return Kronecker(
        B.cholesky_solve(a.left, b.left),
        B.cholesky_solve(a.right, b.right),
    )
