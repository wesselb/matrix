from typing import Union

import lab as B

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense
from ..tiledblocks import TiledBlocks
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

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
def dense(a: AbstractMatrix):
    raise RuntimeError(f"Cannot convert {a} to dense.")


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


@B.dispatch
def dense(a: TiledBlocks):
    if a.dense is None:
        repeated_blocks = []
        for block, rep in zip(a.blocks, a.reps):
            if a.axis == 0:
                repeated_blocks.append(B.tile(block, rep, 1))
            elif a.axis == 1:
                repeated_blocks.append(B.tile(block, 1, rep))
            else:
                raise RuntimeError(f"Invalid axis {a.axis}.")
        a.dense = B.concat(*repeated_blocks, axis=a.axis)
    return a.dense
