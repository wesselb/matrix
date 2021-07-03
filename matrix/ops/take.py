import lab as B
import numpy as np
from lab.util import resolve_axis
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..util import ToDenseWarning

__all__ = []


_dispatch = Dispatcher()


def _resolve_arguments(a, indices_or_mask, axis):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    if isinstance(indices_or_mask, (tuple, list)):
        indices_or_mask = np.array(indices_or_mask)
    return indices_or_mask, resolve_axis(a, axis)


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
def take(a: AbstractMatrix, indices_or_mask, axis=0):
    if structured(a):
        warn_upmodule(f"Taking from {a}: converting to dense.", category=ToDenseWarning)
    return Dense(B.take(B.dense(a), indices_or_mask, axis=axis))


@B.dispatch
def take(a: Zero, indices_or_mask, axis=0):
    indices_or_mask, axis = _resolve_arguments(a, indices_or_mask, axis)
    count = _count_indices_or_mask(indices_or_mask)
    if axis == 0:
        return Zero(a.dtype, count, a.cols)
    elif axis == 1:
        return Zero(a.dtype, a.rows, count)
    else:  # pragma: no cover
        # This can never be reached.
        raise ValueError(f"Invalid axis {axis}.")


@B.dispatch
def take(a: Constant, indices_or_mask, axis=0):
    indices_or_mask, axis = _resolve_arguments(a, indices_or_mask, axis)
    count = _count_indices_or_mask(indices_or_mask)
    if axis == 0:
        return Constant(a.const, count, a.cols)
    elif axis == 1:
        return Constant(a.const, a.rows, count)
    else:  # pragma: no cover
        # This can never be reached.
        raise ValueError(f"Invalid axis {axis}.")


@B.dispatch
def take(a: LowRank, indices_or_mask, axis=0):
    axis = resolve_axis(a, axis)
    if axis == 0:
        return LowRank(B.take(a.left, indices_or_mask), a.right, a.middle)
    elif axis == 1:
        return LowRank(a.left, B.take(a.right, indices_or_mask), a.middle)
    else:  # pragma: no cover
        # This can never be reached.
        raise ValueError(f"Invalid axis {axis}.")
