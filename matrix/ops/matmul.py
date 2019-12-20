from algebra import proven
import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense
from ..shape import assert_compatible, broadcast
from ..woodbury import Woodbury

__all__ = []


def _tr(a, do):
    return B.transpose(a) if do else a


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    assert_compatible(a, b)
    return _tr(b, tr_b)


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    assert_compatible(a, b)
    return _tr(a, tr_a)


@B.dispatch(Dense, Dense)
def matmul(a, b, tr_a=False, tr_b=False):
    return Dense(B.matmul(a.mat, b.mat, tr_a=tr_a, tr_b=tr_b))
