from typing import Union

import lab as B
from plum import convert

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, AbstractMatrix
from ..shape import broadcast
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def shape_matrix(a: B.Numeric):
    return B.shape_matrix(convert(a, AbstractMatrix))


@B.dispatch
def shape_matrix(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.shape(a.mat)[-2:]


@B.dispatch
def shape_matrix(a: Diagonal):
    diag_len = B.shape(a.diag)[-1]
    return diag_len, diag_len


@B.dispatch
def shape_matrix(a: Union[Zero, Constant]):
    return a.rows, a.cols


@B.dispatch
def shape_matrix(a: LowRank):
    return B.shape_matrix(a.left)[0], B.shape_matrix(a.right)[0]


@B.dispatch
def shape_matrix(a: Woodbury):
    return broadcast(B.shape_matrix(a.diag), B.shape_matrix(a.lr))


@B.dispatch
def shape_matrix(a: Kronecker):
    left_rows, left_cols = B.shape_matrix(a.left)
    right_rows, right_cols = B.shape_matrix(a.right)
    return left_rows * right_rows, left_cols * right_cols


@B.dispatch
def shape(a: TiledBlocks):
    if a.axis == 0:
        cols = B.shape(a.blocks[0], 1)
        rows = 0
        for block, rep in zip(a.blocks, a.reps):
            rows += rep * B.shape(block, 0)
        return rows, cols
    elif a.axis == 1:
        rows = B.shape(a.blocks[0], 0)
        cols = 0
        for block, rep in zip(a.blocks, a.reps):
            cols += rep * B.shape(block, 1)
        return rows, cols
    else:
        raise RuntimeError(f"Invalid axis {a.axis}.")


B.shape_matrix = shape_matrix
