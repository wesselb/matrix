import lab as B

# noinspection PyUnresolvedReferences
from ..util import (
    approx,
    check_un_op,
    zero_r,
    dense_r,
    diag1,
    const_r,
    lt1,
    ut1,
    lr1,
    lr_r,
    wb1,
    kron_r,
)


def test_shape_zero(zero_r):
    check_un_op(B.shape, zero_r, asserted_type=tuple)


def test_shape_dense(dense_r):
    check_un_op(B.shape, dense_r, asserted_type=tuple)


def test_shape_diag(diag1):
    check_un_op(B.shape, diag1, asserted_type=tuple)


def test_shape_const(const_r):
    check_un_op(B.shape, const_r, asserted_type=tuple)


def test_shape_lt(lt1):
    check_un_op(B.shape, lt1, asserted_type=tuple)


def test_shape_ut(ut1):
    check_un_op(B.shape, ut1, asserted_type=tuple)


def test_shape_lr(lr_r):
    check_un_op(B.shape, lr_r, asserted_type=tuple)


def test_shape_wb(wb1):
    check_un_op(B.shape, wb1, asserted_type=tuple)


def test_shape_kron(kron_r):
    check_un_op(B.shape, kron_r, asserted_type=tuple)
