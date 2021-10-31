import logging
from typing import Union

import lab as B
from algebra import proven
from plum import convert
from wbml.warning import warn_upmodule

from .algorithms.align import align
from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import assert_compatible
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []

log = logging.getLogger(__name__)


def _reverse_call(t0, t1):
    @B.add.dispatch
    def add(a: t1, b: t0):
        return add(b, a)


# Zero


@B.dispatch(precedence=proven())
def add(a: AbstractMatrix, b: Zero):
    assert_compatible(B.shape(a), B.shape(b))
    return B.broadcast_to(a, *B.shape_broadcast(a, b))


@B.dispatch(precedence=proven())
def add(a: Zero, b: AbstractMatrix):
    assert_compatible(B.shape(a), B.shape(b))
    return B.broadcast_to(b, *B.shape_broadcast(a, b))


# Dense


@B.dispatch
def add(a: AbstractMatrix, b: AbstractMatrix):
    if structured(a) and structured(b):
        warn_upmodule(
            f"Adding {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.add(B.dense(a), B.dense(b)))


@B.dispatch
def add(a: Dense, b: Dense):
    return Dense(B.add(a.mat, b.mat))


# Diagonal


@B.dispatch
def add(a: Diagonal, b: Diagonal):
    return Diagonal(B.add(a.diag, b.diag))


# Constant


@B.dispatch
def add(a: Constant, b: Constant):
    assert_compatible(B.shape(a), B.shape(b))
    return Constant(B.add(a.const, b.const), *B.shape_broadcast(a, b))


@B.dispatch
def add(a: Constant, b: AbstractMatrix):
    if structured(b):
        warn_upmodule(
            f"Adding {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(
        B.broadcast_to(
            B.add(
                B.expand_dims(a.const, axis=-1, times=2, ignore_scalar=True),
                B.dense(b),
            ),
            *B.shape_broadcast(a, b),
        )
    )


@B.dispatch
def add(a: Constant, b: Diagonal):
    assert_compatible(B.shape(a), B.shape(b))
    a = Constant(a.const, *B.shape_matrix_broadcast(a, b))
    return add(convert(a, LowRank), b)


_reverse_call(Constant, AbstractMatrix)
_reverse_call(Constant, Diagonal)


# LowerTriangular


@B.dispatch
def add(a: LowerTriangular, b: LowerTriangular):
    return LowerTriangular(a.mat + b.mat)


@B.dispatch
def add(a: LowerTriangular, b: Diagonal):
    # TODO: Optimise away `B.dense` call.
    return LowerTriangular(a.mat + B.dense(b))


_reverse_call(LowerTriangular, Diagonal)


# UpperTriangular


@B.dispatch
def add(a: UpperTriangular, b: UpperTriangular):
    return UpperTriangular(a.mat + b.mat)


@B.dispatch
def add(a: UpperTriangular, b: LowerTriangular):
    return Dense(a.mat + b.mat)


@B.dispatch
def add(a: UpperTriangular, b: Diagonal):
    return UpperTriangular(a.mat + B.dense(b))


_reverse_call(UpperTriangular, Diagonal)
_reverse_call(UpperTriangular, LowerTriangular)


# LowRank


def _pad_zero_row(a):
    zeros = B.zeros(B.dtype(a), *B.shape_batch(a), 1, B.shape(a, -1))
    return B.concat(a, zeros, axis=-2)


def _pad_zero_col(a):
    zeros = B.zeros(B.dtype(a), *B.shape_batch(a), B.shape(a, -2), 1)
    return B.concat(a, zeros, axis=-1)


def _pad_zero_both(a):
    return _pad_zero_col(_pad_zero_row(a))


@B.dispatch
def add(a: LowRank, b: LowRank):
    assert_compatible(B.shape(a), B.shape(b))

    l_a_p, l_b_p, l_a_jp, l_b_jp = align(a.left, b.left)
    r_a_p, r_b_p, r_a_jp, r_b_jp = align(a.right, b.right)

    # Join left parts.
    a_left = B.take(_pad_zero_col(a.left), l_a_jp, axis=-1)
    b_left = B.take(_pad_zero_col(b.left), l_b_jp, axis=-1)
    join_left = add(a_left, b_left)

    # Join right parts.
    a_right = B.take(_pad_zero_col(a.right), r_a_jp, axis=-1)
    b_right = B.take(_pad_zero_col(b.right), r_b_jp, axis=-1)
    join_right = add(a_right, b_right)

    # Join middle parts.
    a_mid = B.take(B.take(_pad_zero_both(a.middle), l_a_p, axis=-2), r_a_p, axis=-1)
    b_mid = B.take(B.take(_pad_zero_both(b.middle), l_b_p, axis=-2), r_b_p, axis=-1)
    join_middle = add(a_mid, b_mid)

    # Construct resulting low-rank matrix.
    lr = LowRank(join_left, join_right, join_middle)

    # The low-rank matrix may be zero. Check this.
    if B.control_flow.use_cache:
        is_zero = B.control_flow.get_outcome("add:is_zero")
    else:
        is_zero = B.mean(B.abs(join_middle)) < 1e-10
        if B.control_flow.caching:
            B.control_flow.set_outcome("add:is_zero", is_zero)

    # Return zero if the low-rank matrix is zero and return the low-rank matrix
    # otherwise.
    if is_zero:
        return B.zeros(lr)
    else:
        return lr


@B.dispatch
def add(a: LowRank, b: Constant):
    b = Constant(b.const, *B.shape_matrix_broadcast(a, b))
    return add(a, convert(b, LowRank))


@B.dispatch
def add(a: Constant, b: LowRank):
    return add(b, a)


@B.dispatch
def add(a: LowRank, b: Diagonal):
    return Woodbury(b, a)


@B.dispatch
def add(a: Diagonal, b: LowRank):
    return Woodbury(a, b)


# Woodbury


@B.dispatch
def add(a: Woodbury, b: Woodbury):
    return B.add(add(a.diag, b.diag), add(a.lr, b.lr))


@B.dispatch
def add(a: Woodbury, b: Diagonal):
    return B.add(add(a.diag, b), a.lr)


@B.dispatch
def add(a: Diagonal, b: Woodbury):
    return add(b, a)


@B.dispatch.multi((Woodbury, Constant), (Woodbury, LowRank))
def add(a: Woodbury, b: Union[Constant, LowRank]):
    return B.add(a.diag, add(a.lr, b))


@B.dispatch.multi((Constant, Woodbury), (LowRank, Woodbury))
def add(a: Union[Constant, LowRank], b: Woodbury):
    return add(b, a)
