import lab as B
from lab.util import resolve_axis
from plum import Union
from wbml.warning import warn_upmodule

from .util import align_batch, batch_ones
from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def sum(a: Zero, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        return B.zero(a.dtype)
    elif resolved_axis == -2:
        return B.zeros(a.dtype, *B.shape_batch(a), a.cols)
    elif resolved_axis == -1:
        return B.zeros(a.dtype, *B.shape_batch(a), a.rows)
    else:
        batch = list(B.shape_batch(a))
        # Make the axis positive: `batch` has a different number of elements.
        del batch[resolve_axis(a, axis)]
        return B.zeros(a.dtype, *batch, a.rows, a.cols)


@B.dispatch
def sum(
    a: Union[Dense, LowerTriangular, UpperTriangular],
    axis: Union[B.Int, None] = None,
):
    return B.sum(a.mat, axis=axis)


@B.dispatch
def sum(a: Diagonal, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        return B.sum(a.diag)
    elif resolved_axis in {-2, -1}:
        return a.diag
    else:
        # Make the axis positive: `a.diag` has a different rank.
        return Diagonal(B.sum(a.diag, axis=resolve_axis(a, axis)))


@B.dispatch
def sum(a: Constant, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        return B.sum(a.const) * a.rows * a.cols
    elif resolved_axis == -2:
        const = B.expand_dims(a.const, axis=-1, ignore_scalar=True)
        return const * a.rows * B.ones(B.dtype(a.const), *batch_ones(a), a.cols)
    elif resolved_axis == -1:
        const = B.expand_dims(a.const, axis=-1, ignore_scalar=True)
        return const * a.cols * B.ones(B.dtype(a.const), *batch_ones(a), a.rows)
    else:
        # Make the axis positive: `a.const` has a different rank.
        return Constant(B.sum(a.const, axis=resolve_axis(a, axis)), a.rows, a.cols)


@B.dispatch
def sum(a: LowRank, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        return B.sum(
            B.sum(B.matmul(a.left, a.middle), axis=-2) * B.sum(a.right, axis=-2)
        )
    elif resolved_axis == -2:
        return B.sum(
            B.multiply(
                B.expand_dims(B.sum(a.left, axis=-2), axis=-2),
                B.matmul(a.right, a.middle, tr_b=True),
            ),
            axis=-1,
        )
    elif resolved_axis == -1:
        return B.sum(
            B.multiply(
                B.matmul(a.left, a.middle),
                B.expand_dims(B.sum(a.right, axis=-2), axis=-2),
            ),
            axis=-1,
        )
    else:
        warn_upmodule(
            f"Summing {a} over batch dimensions: converting to dense.",
            category=ToDenseWarning,
        )
        return B.sum(B.dense(a), axis=axis)


@B.dispatch
def sum(a: Woodbury, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        # We must sum in two steps to get broadcasting right!
        diag_sum = B.sum(B.sum(a.diag, axis=-1), axis=-1)
        lr_sum = B.sum(B.sum(a.lr, axis=-1), axis=-1)
        return B.sum(lr_sum + diag_sum)
    elif resolved_axis in {-1, -2}:
        return B.sum(a.diag, axis=resolved_axis) + B.sum(a.lr, axis=resolved_axis)
    else:
        # We're summing over batch dimensions: align the batches first.
        diag, lr = align_batch(a.diag, a.lr)
        axis = resolve_axis(a, axis)
        return B.sum(diag, axis=axis) + B.sum(lr, axis=axis)


@B.dispatch
def sum(a: Kronecker, axis: Union[B.Int, None] = None):
    resolved_axis = resolve_axis(a, axis, negative=True)
    if resolved_axis is None:
        left_sum = B.sum(B.sum(a.left, axis=-1), axis=-1)
        right_sum = B.sum(B.sum(a.right, axis=-1), axis=-1)
        return B.sum(left_sum * right_sum)
    elif resolved_axis == -2:
        left_sum = B.sum(a.left, axis=-2)[..., :, None]
        right_sum = B.sum(a.right, axis=-2)[..., None, :]
        return B.reshape(left_sum * right_sum, *B.shape_batch(a), B.shape_matrix(a, 1))
    elif resolved_axis == -1:
        left_sum = B.sum(a.left, axis=-1)[..., :, None]
        right_sum = B.sum(a.right, axis=-1)[..., None, :]
        return B.reshape(left_sum * right_sum, *B.shape_batch(a), B.shape_matrix(a, 0))
    else:
        warn_upmodule(
            f"Summing {a} over batch dimensions: converting to dense.",
            category=ToDenseWarning,
        )
        return B.sum(B.dense(a), axis=axis)
