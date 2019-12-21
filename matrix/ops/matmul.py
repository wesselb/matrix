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
    assert s1[1] == s2[0], f'Objects {a} and {b} are asserted to be ' \
                           f'composable, but they are not.'


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
    return LowRank(a.const * ones, B.expand_dims(B.sum(b, axis=0), axis=1))


@B.dispatch(AbstractMatrix, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    ones = B.ones(B.dtype(b), b.cols, 1)
    return LowRank(B.expand_dims(B.sum(a, axis=1), axis=1), b.const * ones)


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


def _reshape_cols(a, *indices):
    return B.transpose(B.reshape(B.transpose(a), *reversed(indices)))


_reshape_rows = B.reshape


def _kron_id_a_x(a, b):
    reshaped = _reshape_cols(b, B.shape(a)[1], -1)
    return _reshape_cols(B.matmul(a, reshaped), -1, B.shape(b)[1])


def _kron_a_id_x(a, b):
    reshaped = _reshape_rows(b, B.shape(a)[1], -1)
    return _reshape_rows(B.matmul(a, reshaped), -1, B.shape(b)[1])


@B.dispatch(Kronecker, Dense)
def matmul(a, b, tr_a=False, tr_b=False):
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)

    # Extract the factors of the product.
    left = a.left.mat
    right = a.right.mat
    l_rows, l_cols = B.shape(left)
    r_rows, r_cols = B.shape(right)

    cost_first_left = l_rows * l_cols * r_cols + r_rows * r_cols * l_rows
    cost_first_right = r_rows * r_cols * l_cols + l_rows * l_cols * r_rows

    if cost_first_left <= cost_first_right:
        return Dense(_kron_id_a_x(right, _kron_a_id_x(left, b.mat)))
    else:
        return Dense(_kron_a_id_x(left, _kron_id_a_x(right, b.mat)))


@B.dispatch(Dense, Kronecker)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.transpose(B.matmul(b, a, tr_a=not tr_b, tr_b=not tr_a))
