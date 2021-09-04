import lab as B
import numpy as np
from lab.util import resolve_axis
from plum import Dispatcher
from wbml.warning import warn_upmodule

from .util import align_batch
from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


_dispatch = Dispatcher()


@B.dispatch
def take(a: AbstractMatrix, indices_or_mask, axis: B.Int = -2):
    if structured(a):
        warn_upmodule(f"Taking from {a}: converting to dense.", category=ToDenseWarning)
    return Dense(B.take(B.dense(a), indices_or_mask, axis=axis))


def _resolve_arguments(a, indices_or_mask, axis):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    if isinstance(indices_or_mask, (tuple, list)):
        indices_or_mask = np.array(indices_or_mask)
    return indices_or_mask, resolve_axis(a, axis, negative=True)


@_dispatch
def _count_indices_or_mask(indices_or_mask: B.Numeric):
    mask = B.issubdtype(B.dtype(indices_or_mask), bool)
    if mask:
        # Count all `True`s.
        return B.sum(B.jit_to_numpy(indices_or_mask))
    else:
        # Count the number of indices.
        return B.shape(indices_or_mask, 0)


@B.dispatch
def take(a: Zero, indices_or_mask, axis: B.Int = -2):
    indices_or_mask, axis = _resolve_arguments(a, indices_or_mask, axis)
    count = _count_indices_or_mask(indices_or_mask)
    shape = list(B.shape(a))
    shape[axis] = count
    return Zero(a.dtype, *shape)


@B.dispatch
def take(a: Diagonal, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    if axis in {-2, -1}:
        return take.invoke(AbstractMatrix, object)(a, indices_or_mask, axis)
    else:
        return Diagonal(B.take(a.diag, indices_or_mask, axis=resolve_axis(a, axis)))


@B.dispatch
def take(a: Constant, indices_or_mask, axis: B.Int = -2):
    indices_or_mask, axis = _resolve_arguments(a, indices_or_mask, axis)
    count = _count_indices_or_mask(indices_or_mask)
    if axis == -2:
        return Constant(a.const, count, a.cols)
    elif axis == -1:
        return Constant(a.const, a.rows, count)
    else:
        return Constant(
            B.take(a.const, indices_or_mask, axis=resolve_axis(a, axis)),
            a.rows,
            a.cols,
        )


@B.dispatch
def take(a: LowerTriangular, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    mat_taken = B.take(a.mat, indices_or_mask, axis=axis)
    if axis in {-2, -1}:
        return Dense(mat_taken)
    else:
        return LowerTriangular(mat_taken)


@B.dispatch
def take(a: UpperTriangular, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    mat_taken = B.take(a.mat, indices_or_mask, axis=axis)
    if axis in {-2, -1}:
        return Dense(mat_taken)
    else:
        return UpperTriangular(mat_taken)


@B.dispatch
def take(a: LowRank, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    if axis == -2:
        return LowRank(B.take(a.left, indices_or_mask), a.right, a.middle)
    elif axis == -1:
        return LowRank(a.left, B.take(a.right, indices_or_mask), a.middle)
    else:
        left, middle, right = align_batch(a.left, a.middle, a.right)
        axis = resolve_axis(a, axis)
        return LowRank(
            B.take(left, indices_or_mask, axis=axis),
            B.take(right, indices_or_mask, axis=axis),
            middle=B.take(middle, indices_or_mask, axis=axis),
        )


@B.dispatch
def take(a: Woodbury, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    if axis in {-2, -1}:
        diag = B.take(a.diag, indices_or_mask, axis=axis)
        lr = B.take(a.lr, indices_or_mask, axis=axis)
        return diag + lr
    else:
        diag, lr = align_batch(a.diag, a.lr)
        axis = resolve_axis(a, axis)
        diag = B.take(diag, indices_or_mask, axis=axis)
        lr = B.take(lr, indices_or_mask, axis=axis)
        return diag + lr


@B.dispatch
def take(a: Kronecker, indices_or_mask, axis: B.Int = -2):
    axis = resolve_axis(a, axis, negative=True)
    if axis in {-2, -1}:
        return take.invoke(AbstractMatrix, object)(a, indices_or_mask, axis)
    else:
        left, right = align_batch(a.left, a.right)
        axis = resolve_axis(a, axis)
        return Kronecker(
            B.take(left, indices_or_mask, axis=axis),
            B.take(right, indices_or_mask, axis=axis),
        )
