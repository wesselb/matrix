import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..matrix import Dense
from ..lowrank import LowRank
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury
from ..kronecker import Kronecker
from ..shape import batch_ones

__all__ = []


@B.dispatch(object)
def dense(a):
    """Convert a (structured) matrix to a dense, numeric matrix.

    Args:
        a (matrix): Matrix to convert to dense.

    Returns:
        matrix: Dense version of `a`.
    """
    return a


B.dense = dense


@B.dispatch(Zero)
def dense(a):
    if a.dense is None:
        a.dense = B.zeros(a.dtype, *B.shape(a))
    return a.dense


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def dense(a):
    return a.mat


@B.dispatch(Diagonal)
def dense(a):
    if a.dense is None:
        a.dense = B.diag_construct(a.diag)
    return a.dense


@B.dispatch(Constant)
def dense(a):
    if a.dense is None:
        ones_shape = batch_ones(a) + B.shape_matrix(a)
        const = B.expand_dims(B.expand_dims(a.const, axis=-1), axis=-1)
        a.dense = const * B.ones(B.dtype(a.const), *ones_shape)
    return a.dense


@B.dispatch(LowRank)
def dense(a):
    if a.dense is None:
        a.dense = B.dense(B.mm(a.left, a.middle, a.right, tr_c=True))
    return a.dense


@B.dispatch(Woodbury)
def dense(a):
    if a.dense is None:
        a.dense = B.dense(a.diag) + B.dense(a.lr)
    return a.dense


@B.dispatch(Kronecker)
def dense(a):
    if a.dense is None:
        left = B.dense(a.left)
        right = B.dense(a.right)
        # Compute the Kronecker products whilst enabling batching.
        left = left[..., :, None, :, None]
        right = right[..., None, :, None, :]
        # Trust that the shape computation works correctly.
        a.dense = B.reshape(left * right, *B.shape(a))
    return a.dense
