from typing import Union

import lab as B

from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix
from ..shape import assert_square
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def logdet(a: AbstractMatrix):
    return 2 * B.logdet(B.cholesky(a))


@B.dispatch
def logdet(a: Union[Diagonal, LowerTriangular, UpperTriangular]):
    return B.sum(B.log(B.diag(a)), axis=-1)


@B.dispatch
def logdet(a: Woodbury):
    return B.logdet(a.diag) + B.logdet(a.lr.middle) + B.logdet(B.pd_schur(a))


@B.dispatch
def logdet(a: Kronecker):
    assert_square(
        a.left, f"Left factor of {a} must be square to compute the log-determinant."
    )
    assert_square(
        a.right,
        f"Right factor of {a} must be square to compute the log-determinant.",
    )
    n = B.shape_matrix(a.left, 0)
    m = B.shape_matrix(a.right, 0)
    return m * B.logdet(a.left) + n * B.logdet(a.right)
