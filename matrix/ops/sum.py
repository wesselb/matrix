import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..shape import matrix_axis, batch_ones
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


def _raise(axis):
    raise ValueError(f"Cannot sum over axis {axis}.")


@B.dispatch(Zero)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        return B.zero(a.dtype)
    elif resolved_axis == 0:
        return B.zeros(a.dtype, *B.shape_batch(a), a.cols)
    elif resolved_axis == 1:
        return B.zeros(a.dtype, *B.shape_batch(a), a.rows)
    else:
        _raise(axis)


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def sum(a, axis=None):
    return B.sum(a.mat, axis=axis)


@B.dispatch(Diagonal)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        return B.sum(a.diag)
    elif resolved_axis == 0 or resolved_axis == 1:
        return a.diag
    else:
        _raise(axis)


@B.dispatch(Constant)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        return B.sum(a.const) * a.rows * a.cols
    elif resolved_axis == 0:
        const = B.expand_dims(a.const, axis=-1)
        return const * a.rows * B.ones(B.dtype(a.const), *batch_ones(a), a.cols)
    elif resolved_axis == 1:
        const = B.expand_dims(a.const, axis=-1)
        return const * a.cols * B.ones(B.dtype(a.const), *batch_ones(a), a.rows)
    else:
        _raise(axis)


@B.dispatch(LowRank)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        return B.sum(
            B.sum(B.matmul(a.left, a.middle), axis=-2) * B.sum(a.right, axis=-2)
        )
    elif resolved_axis == 0:
        return B.sum(
            B.multiply(
                B.expand_dims(B.sum(a.left, axis=-2), axis=-2),
                B.matmul(a.right, a.middle, tr_b=True),
            ),
            axis=-1,
        )
    elif resolved_axis == 1:
        return B.sum(
            B.multiply(
                B.matmul(a.left, a.middle),
                B.expand_dims(B.sum(a.right, axis=-2), axis=-2),
            ),
            axis=-1,
        )
    else:
        _raise(axis)


@B.dispatch(Woodbury)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        lr_sum = B.sum(B.sum(a.lr, axis=-1), axis=-1)
        diag_sum = B.sum(B.sum(a.diag, axis=-1), axis=-1)
        return B.sum(lr_sum + diag_sum)
    elif resolved_axis == 0:
        return B.sum(a.lr, axis=-2) + B.sum(a.diag, axis=-2)
    elif resolved_axis == 1:
        return B.sum(a.lr, axis=-1) + B.sum(a.diag, axis=-1)
    else:
        _raise(axis)


@B.dispatch(Kronecker)
def sum(a, axis=None):
    resolved_axis = matrix_axis(a, axis)
    if resolved_axis is None:
        left_sum = B.sum(B.sum(a.left, axis=-1), axis=-1)
        right_sum = B.sum(B.sum(a.right, axis=-1), axis=-1)
        return B.sum(left_sum * right_sum)
    elif resolved_axis == 0:
        left_sum = B.sum(a.left, axis=-2)[..., :, None]
        right_sum = B.sum(a.right, axis=-2)[..., None, :]
        return B.reshape(left_sum * right_sum, *B.shape_batch(a), B.shape_matrix(a)[1])
    elif resolved_axis == 1:
        left_sum = B.sum(a.left, axis=-1)[..., :, None]
        right_sum = B.sum(a.right, axis=-1)[..., None, :]
        return B.reshape(left_sum * right_sum, *B.shape_batch(a), B.shape_matrix(a)[0])
    else:
        _raise(axis)
