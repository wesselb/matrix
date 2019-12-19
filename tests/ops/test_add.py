import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,

    zero1,
    zero2,
    dense1,
    dense2,
    diag1,
    diag2,
    const_or_scalar1,
    const_or_scalar2,
    const1,
    const2,
    lr1,
    lr2,
    wb1,
    wb2
)


def test_add_zero_diag(zero1, diag2):
    check_bin_op(B.add, zero1, diag2, asserted_type=Diagonal)
    check_bin_op(B.add, diag2, zero1, asserted_type=Diagonal)


def test_add_dense(dense1, dense2):
    check_bin_op(B.add, dense1, dense2, asserted_type=Dense)


def test_add_diag_dense(diag1, dense2):
    check_bin_op(B.add, diag1, dense2, asserted_type=Dense)


def test_add_diag(diag1, diag2):
    check_bin_op(B.add, diag1, diag2, asserted_type=Diagonal)


def test_add_const_dense(const_or_scalar1, dense2):
    check_bin_op(B.add, const_or_scalar1, dense2, asserted_type=Dense)
    check_bin_op(B.add, dense2, const_or_scalar1, asserted_type=Dense)


def test_add_const_diag(const_or_scalar1, diag2):
    check_bin_op(B.add, const_or_scalar1, diag2, asserted_type=Dense)


def test_add_const(const_or_scalar1, const2):
    check_bin_op(B.add, const_or_scalar1, const2, asserted_type=Constant)


def test_add_lr(lr1, lr2):
    check_bin_op(B.add, lr1, lr2, asserted_type=LowRank)


def test_add_const_lr(const_or_scalar1, lr2):
    check_bin_op(B.add, const_or_scalar1, lr2, asserted_type=LowRank)
    check_bin_op(B.add, lr2, const_or_scalar1, asserted_type=LowRank)


def test_add_diag_lr(diag1, lr2):
    check_bin_op(B.add, diag1, lr2, asserted_type=Woodbury)
    check_bin_op(B.add, lr2, diag1, asserted_type=Woodbury)


def test_add_wb(wb1, wb2):
    check_bin_op(B.add, wb1, wb2, asserted_type=Woodbury)


def test_add_wb_diag(wb1, diag1):
    check_bin_op(B.add, wb1, diag1, asserted_type=Woodbury)
    check_bin_op(B.add, diag1, wb1, asserted_type=Woodbury)


def test_add_wb_constant(wb1, const_or_scalar2):
    check_bin_op(B.add, wb1, const_or_scalar2, asserted_type=Woodbury)
    check_bin_op(B.add, const_or_scalar2, wb1, asserted_type=Woodbury)


def test_add_wb_lr(wb1, lr2):
    check_bin_op(B.add, wb1, lr2, asserted_type=Woodbury)
    check_bin_op(B.add, lr2, wb1, asserted_type=Woodbury)
