import lab as B
import numpy as np
from plum import Dispatcher
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury
from .take import _count_indices_or_mask

__all__ = []

_dispatch = Dispatcher()


def _resolve_indices_or_mask(indices_or_mask):
    if B.rank(indices_or_mask) != 1:
        raise ValueError("Indices or mask must be rank 1.")
    if isinstance(indices_or_mask, (tuple, list)):
        indices_or_mask = np.array(indices_or_mask)
    return indices_or_mask


@B.dispatch
def submatrix(a: AbstractMatrix, indices_or_mask):
    if structured(a):
        warn_upmodule(
            f"Taking a submatrix from {a}: converting to dense.",
            category=ToDenseWarning,
        )
    iom = indices_or_mask
    return Dense(B.take(B.take(B.dense(a), iom, axis=-2), iom, axis=-1))


@B.dispatch
def submatrix(a: Zero, indices_or_mask):
    indices_or_mask = _resolve_indices_or_mask(indices_or_mask)
    count = _count_indices_or_mask(indices_or_mask)
    return Zero(a.dtype, *a.batch, count, count)


@B.dispatch
def submatrix(a: Diagonal, indices_or_mask):
    return Diagonal(B.take(a.diag, indices_or_mask, axis=-1))


@B.dispatch
def submatrix(a: Constant, indices_or_mask):
    indices_or_mask = _resolve_indices_or_mask(indices_or_mask)
    count = _count_indices_or_mask(indices_or_mask)
    return Constant(a.const, *a.batch, count, count)


@B.dispatch
def submatrix(a: LowerTriangular, indices_or_mask):
    return LowerTriangular(B.submatrix(a.mat, indices_or_mask))


@B.dispatch
def submatrix(a: UpperTriangular, indices_or_mask):
    return UpperTriangular(B.submatrix(a.mat, indices_or_mask))


@B.dispatch
def submatrix(a: LowRank, indices_or_mask):
    return LowRank(
        B.take(a.left, indices_or_mask, axis=-2),
        B.take(a.right, indices_or_mask, axis=-2),
        a.middle,
    )


@B.dispatch
def submatrix(a: Woodbury, indices_or_mask):
    return Woodbury(
        B.submatrix(a.diag, indices_or_mask),
        B.submatrix(a.lr, indices_or_mask),
    )
