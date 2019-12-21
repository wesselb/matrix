from algebra import proven
import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense
from ..shape import assert_compatible, get_shape
from ..woodbury import Woodbury

__all__ = []


def _tr(a, do):
    return B.transpose(a) if do else a


def _assert_composable(a, b, tr_a=False, tr_b=False):
    s1, s2 = get_shape(a, b)
    s1 = _tr(s1, tr_a)
    s2 = _tr(s2, tr_b)
    assert_compatible(s1[1], s2[0])


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return _tr(b, tr_b)


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return _tr(a, tr_a)


@B.dispatch(Dense, Dense)
def matmul(a, b, tr_a=False, tr_b=False):
    return Dense(B.matmul(a.mat, b.mat, tr_a=tr_a, tr_b=tr_b))


@B.dispatch(Diagonal, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return Diagonal(B.multiply(a.diag, b.diag))


@B.dispatch(Diagonal, Dense)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return Dense(a.diag[:, None] * _tr(b.mat, tr_b))


@B.dispatch(Dense, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return Dense(_tr(a.mat, tr_a) * b.diag[None, :])
