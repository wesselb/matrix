from typing import Union

import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def shape(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.shape(a.mat)


@B.dispatch
def shape(a: Diagonal):
    diag_len = B.shape(a.diag)[0]
    return diag_len, diag_len


@B.dispatch
def shape(a: Union[Zero, Constant]):
    return a.rows, a.cols


@B.dispatch
def shape(a: LowRank):
    return B.shape(a.left)[0], B.shape(a.right)[0]


@B.dispatch
def shape(a: Woodbury):
    return B.shape(a.lr)


@B.dispatch
def shape(a: Kronecker):
    left_rows, left_cols = B.shape(a.left)
    right_rows, right_cols = B.shape(a.right)
    return left_rows * right_rows, left_cols * right_cols
