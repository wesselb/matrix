import pytest
import lab as B

from matrix import (
    AbstractMatrix,
    Dense,
    Diagonal,
    Zero,
    Constant,
    LowRank,
    Woodbury,
    Kronecker
)
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_bin_op,
    AssertDenseWarning,

    zero1,
    zero2,
    dense1,
    dense2,
    dense_bc,
    diag1,
    diag2,
    const_or_scalar1,
    const_or_scalar2,
    const1,
    const2,
    lr1,
    lr2,
    wb1,
    wb2,
    kron1,
    kron2
)


def test_multiply_zero_diag(zero1, diag2):
    check_bin_op(B.multiply, zero1, diag2, asserted_type=Zero)
    check_bin_op(B.multiply, diag2, zero1, asserted_type=Zero)


def test_multiply_dense(dense1, dense2):
    check_bin_op(B.multiply, dense1, dense2, asserted_type=Dense)


def test_multiply_diag(diag1, diag2):
    check_bin_op(B.multiply, diag1, diag2, asserted_type=Diagonal)


def test_multiply_diag_dense(diag1, dense_bc):
    check_bin_op(B.multiply, diag1, dense_bc, asserted_type=Diagonal)
    check_bin_op(B.multiply, dense_bc, diag1, asserted_type=Diagonal)


def test_multiply_const(const_or_scalar1, const2):
    check_bin_op(B.multiply, const_or_scalar1, const2, asserted_type=Constant)


def test_multiply_const_dense(const_or_scalar1, dense2):
    check_bin_op(B.multiply, const_or_scalar1, dense2, asserted_type=Dense)
    check_bin_op(B.multiply, dense2, const_or_scalar1, asserted_type=Dense)


def test_multiply_const_diag(const_or_scalar1, diag2):
    check_bin_op(B.multiply, const_or_scalar1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, const_or_scalar1, asserted_type=Diagonal)


def test_multiply_lr(lr1, lr2):
    check_bin_op(B.multiply, lr1, lr2, asserted_type=LowRank)


def test_multiply_lr_const(lr1, const_or_scalar2):
    check_bin_op(B.multiply, lr1, const_or_scalar2, asserted_type=LowRank)
    check_bin_op(B.multiply, const_or_scalar2, lr1, asserted_type=LowRank)


def test_multiply_lr_diag(lr1, diag2):
    check_bin_op(B.multiply, lr1, diag2, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag2, lr1, asserted_type=Diagonal)


def test_multiply_wb(wb1, wb2):
    check_bin_op(B.multiply, wb1, wb2, asserted_type=Woodbury)


def test_multiply_wb_diag(wb1, diag1):
    check_bin_op(B.multiply, wb1, diag1, asserted_type=Diagonal)
    check_bin_op(B.multiply, diag1, wb1, asserted_type=Diagonal)


def test_multiply_wb_const(wb1, const_or_scalar2):
    check_bin_op(B.multiply, wb1, const_or_scalar2, asserted_type=Woodbury)
    check_bin_op(B.multiply, const_or_scalar2, wb1, asserted_type=Woodbury)


def test_multiply_wb_lr(wb1, lr2):
    check_bin_op(B.multiply, wb1, lr2, asserted_type=Woodbury)
    check_bin_op(B.multiply, lr2, wb1, asserted_type=Woodbury)


def test_multiply_kron(kron1, kron2):
    if (
            B.shape(kron1.left) == B.shape(kron2.left) and
            B.shape(kron1.right) == B.shape(kron2.right)
    ):
        check_bin_op(B.multiply, kron1, kron2, asserted_type=Kronecker)
    else:
        with pytest.raises(AssertionError):
            B.matmul(kron1, kron2)


def test_multiply_kron_const(kron1, const_or_scalar2):
    check_bin_op(B.multiply, kron1, const_or_scalar2, asserted_type=Kronecker)
    check_bin_op(B.multiply, const_or_scalar2, kron1, asserted_type=Kronecker)
