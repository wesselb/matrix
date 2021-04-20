from typing import Union

import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..matrix import Dense
from ..lowrank import LowRank
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury
from ..kronecker import Kronecker

__all__ = []


@B.dispatch
def dense(a):
    """Convert a (structured) matrix to a dense, numeric matrix.

    Args:
        a (matrix): Matrix to convert to dense.

    Returns:
        matrix: Dense version of `a`.
    """
    return a


B.dense = dense


@B.dispatch
def dense(a: Zero):
    if a.dense is None:
        a.dense = B.zeros(a.dtype, a.rows, a.cols)
    return a.dense


@B.dispatch
def dense(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return a.mat


@B.dispatch
def dense(a: Diagonal):
    if a.dense is None:
        a.dense = B.diag(a.diag)
    return a.dense


@B.dispatch
def dense(a: Constant):
    if a.dense is None:
        a.dense = a.const * B.ones(B.dtype(a.const), a.rows, a.cols)
    return a.dense


@B.dispatch
def dense(a: LowRank):
    if a.dense is None:
        a.dense = B.dense(B.mm(a.left, a.middle, a.right, tr_c=True))
    return a.dense


@B.dispatch
def dense(a: Woodbury):
    if a.dense is None:
        a.dense = B.dense(a.diag) + B.dense(a.lr)
    return a.dense


@B.dispatch
def dense(a: Kronecker):
    if a.dense is None:
        a.dense = B.kron(B.dense(a.left), B.dense(a.right))
    return a.dense
