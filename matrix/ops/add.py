import logging
from typing import Union

import lab as B
from algebra import proven
from plum import convert
from wbml.warning import warn_upmodule

from .algorithms.align import align
from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import assert_compatible, broadcast
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
    assert_compatible(a, b)
    return a


@B.dispatch(precedence=proven())
def add(a: Zero, b: AbstractMatrix):
    assert_compatible(a, b)
    return b


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
    assert_compatible(a, b)
    return Constant(a.const + b.const, *broadcast(a, b).as_tuple())


@B.dispatch
def add(a: Constant, b: AbstractMatrix):
    if structured(b):
        warn_upmodule(
            f"Adding {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(a.const + B.dense(b))


@B.dispatch
def add(a: Constant, b: Diagonal):
    assert_compatible(a, b)
    a = Constant(a.const, *broadcast(a, b).as_tuple())
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
    zeros = B.zeros(B.dtype(a), 1, B.shape(a)[1])
    return B.concat(a, zeros, axis=0)


def _pad_zero_col(a):
    zeros = B.zeros(B.dtype(a), B.shape(a)[0], 1)
    return B.concat(a, zeros, axis=1)


def _pad_zero_both(a):
    return _pad_zero_col(_pad_zero_row(a))


@B.dispatch
def add(a: LowRank, b: LowRank):
    assert_compatible(a, b)

    l_a_p, l_b_p, l_a_jp, l_b_jp = align(a.left, b.left)
    r_a_p, r_b_p, r_a_jp, r_b_jp = align(a.right, b.right)

    # Join left parts.
    a_left = B.take(_pad_zero_col(a.left), l_a_jp, axis=-1)
    b_left = B.take(_pad_zero_col(b.left), l_b_jp, axis=-1)
    join_left = a_left + b_left

    # Join right parts.
    a_right = B.take(_pad_zero_col(a.right), r_a_jp, axis=-1)
    b_right = B.take(_pad_zero_col(b.right), r_b_jp, axis=-1)
    join_right = a_right + b_right

    # Join middle parts.
    a_mid = B.take(B.take(_pad_zero_both(a.middle), l_a_p, axis=-2), r_a_p, axis=-1)
    b_mid = B.take(B.take(_pad_zero_both(b.middle), l_b_p, axis=-2), r_b_p, axis=-1)
    join_middle = a_mid + b_mid

    return LowRank(join_left, join_right, join_middle)


@B.dispatch
def add(a: LowRank, b: Constant):
    assert_compatible(a, b)
    b = Constant(b.const, *broadcast(a, b).as_tuple())
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
    return Woodbury(a.diag + b.diag, a.lr + b.lr)


@B.dispatch
def add(a: Woodbury, b: Diagonal):
    return Woodbury(a.diag + b, a.lr)


@B.dispatch
def add(a: Diagonal, b: Woodbury):
    return add(b, a)


@B.dispatch.multi((Woodbury, Constant), (Woodbury, LowRank))
def add(a: Woodbury, b: Union[Constant, LowRank]):
    return Woodbury(a.diag, a.lr + b)


@B.dispatch.multi((Constant, Woodbury), (LowRank, Woodbury))
def add(a: Union[Constant, LowRank], b: Woodbury):
    return add(b, a)
