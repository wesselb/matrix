import lab as B
import pytest

# noinspection PyUnresolvedReferences
from ..util import (
    AssertDenseWarning,
    approx,
    check_un_op,
    const_r,
    dense_r,
    diag1,
    kron_r,
    lr1,
    lr_r,
    lt1,
    tb1,
    tb_axis,
    ut1,
    wb1,
    zero_r,
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


def test_shape_tb(tb1):
    with AssertDenseWarning(["tiling", "concatenating"]):
        check_un_op(B.shape, tb1, asserted_type=tuple)


def test_shape_tb_axis(tb1):
    tb1.axis = 3
    with pytest.raises(RuntimeError):
        B.shape(tb1)
