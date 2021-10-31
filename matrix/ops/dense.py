from typing import Union

import lab as B
from lab.util import resolve_axis

from .util import align_batch, batch_ones
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
        a.dense = B.zeros(a.dtype, *B.shape(a))
    return a.dense


@B.dispatch
def dense(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return a.mat


@B.dispatch
def dense(a: Diagonal):
    if a.dense is None:
        a.dense = B.diag_construct(a.diag)
    return a.dense


@B.dispatch
def dense(a: Constant):
    if a.dense is None:
        const = B.expand_dims(a.const, axis=-1, times=2, ignore_scalar=True)
        a.dense = const * B.ones(B.dtype(a.const), *batch_ones(a), *B.shape_matrix(a))
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
        left, right = align_batch(a.left, a.right)
        a.dense = B.kron(B.dense(left), B.dense(right), -2, -1)
    return a.dense


@B.dispatch
def dense(a: TiledBlocks):
    if a.dense is None:
        # Take the axis to be negative, which will work when batch dimensions are
        # added during broadcasting.
        axis = resolve_axis(a, a.axis, negative=True)

        # We explicitly broadcast every element to have the right batch shape, because
        # the concatenation will otherwise fail.
        target_batch = B.shape_batch(a)

        repeated_blocks = []
        for block, rep in zip(a.blocks, a.reps):
            # Compute expanded current batch shape of block.
            block_batch = tuple(B.shape_batch(block))
            block_batch = (1,) * (len(target_batch) - len(block_batch)) + block_batch

            # Compute target batch shape of block.
            if axis in {-2, -1}:
                block_target_batch = target_batch
            else:
                block_target_batch = list(target_batch)
                block_target_batch[axis + 2] = block_batch[axis + 2]

            # Broadcast batch shape.
            block = B.broadcast_batch_to(block, *block_target_batch)

            # Perform tiling.
            tile_reps = [1] * (len(target_batch) + 2)
            tile_reps[axis] = rep

            repeated_blocks.append(B.tile(block, *tile_reps))

        a.dense = B.concat(*repeated_blocks, axis=axis)
    return a.dense
