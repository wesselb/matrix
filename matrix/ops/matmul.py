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


@B.dispatch(Constant, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return Constant(a.const * b.const * a.cols, a.rows, b.cols)


@B.dispatch(Constant, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    ones = B.ones(B.dtype(a), a.rows, 1)
    return LowRank(left=a.const * ones,
                   right=B.expand_dims(B.sum(b, axis=0), axis=1))


@B.dispatch(AbstractMatrix, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    ones = B.ones(B.dtype(b), b.cols, 1)
    return LowRank(left=B.expand_dims(B.sum(a, axis=1), axis=1),
                   right=b.const * ones)


@B.dispatch(LowRank, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    middle = B.matmul(a.right, b.left, tr_a=True)
    rows, cols = B.shape(middle)
    if rows > cols:
        return LowRank(B.matmul(a.left, middle), b.right)
    else:
        return LowRank(a.left, B.matmul(b.right, middle, tr_b=True))


@B.dispatch(Diagonal, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    b = _tr(b, tr_b)
    return LowRank(a.diag[:, None] * b.left, b.right)


@B.dispatch(LowRank, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    return LowRank(a.left, a.right * b.diag[:, None])


@B.dispatch.multi((Woodbury, Woodbury),
                  (Woodbury, AbstractMatrix))
def matmul(a, b, tr_a=False, tr_b=False):
    # Expand out Woodbury matrices.
    return B.add(B.matmul(a.diag, b, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a.lr, b, tr_a=tr_a, tr_b=tr_b))


@B.dispatch(AbstractMatrix, Woodbury)
def matmul(a, b, tr_a=False, tr_b=False):
    # Expand out Woodbury matrices.
    return B.add(B.matmul(a, b.diag, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a, b.lr, tr_a=tr_a, tr_b=tr_b))


@B.dispatch(Kronecker, Kronecker)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a.left, b.left, tr_a=tr_a, tr_b=tr_b)
    _assert_composable(a.right, b.right, tr_a=tr_a, tr_b=tr_b)
    return Kronecker(B.matmul(a.left, b.left, tr_a=tr_a, tr_b=tr_b),
                     B.matmul(a.right, b.right, tr_a=tr_a, tr_b=tr_b))
