from typing import Union

import lab as B
from plum import convert

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, AbstractMatrix
from ..shape import broadcast
from ..tiledblocks import TiledBlocks
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def shape_matrix(a, *dims: B.Int):
    """Get the matrix shape of a tensor.

    Args:
        a (tensor): Tensor.
        *dims (int, optional): Dimensions to get.

    Returns:
        object: Matrix shape of `a`.
    """
    a_shape_matrix = B.shape_matrix(a)
    return B.squeeze(tuple(a_shape_matrix[i] for i in dims))


@B.dispatch
def shape_matrix(*elements):
    return broadcast(*(B.shape_matrix(element) for element in elements))


@B.dispatch
def shape_matrix(a: B.Numeric):
    return B.shape_matrix(convert(a, AbstractMatrix))


@B.dispatch
def shape_matrix(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.shape(a.mat)[-2:]


@B.dispatch
def shape_matrix(a: Diagonal):
    diag_len = B.shape(a.diag, -1)
    return diag_len, diag_len


@B.dispatch
def shape_matrix(a: Union[Zero, Constant]):
    return a.rows, a.cols


@B.dispatch
def shape_matrix(a: LowRank):
    return B.shape_matrix(a.left, 0), B.shape_matrix(a.right, 0)


@B.dispatch
def shape_matrix(a: Woodbury):
    return B.shape_matrix(a.diag, a.lr)


@B.dispatch
def shape_matrix(a: Kronecker):
    left_rows, left_cols = B.shape_matrix(a.left)
    right_rows, right_cols = B.shape_matrix(a.right)
    return left_rows * right_rows, left_cols * right_cols


@B.dispatch
def shape_matrix(a: TiledBlocks):
    # `shape` is directly implemented for `TiledBlocks`, and the matrix shape can be
    # inferred from it.
    return B.shape(a)[-2:]


B.shape_matrix = shape_matrix
