import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury, Kronecker, Zero
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    zero_r,
    dense_r,
    diag1,
    const_r,
    lr_r,
    wb1,
    kron_r
)


def test_shorthand(dense_r):
    check_un_op(lambda a: a.T, dense_r, asserted_type=Dense)


def test_transpose_zero(zero_r):
    check_un_op(B.transpose, zero_r, asserted_type=Zero)


def test_transpose_dense(dense_r):
    check_un_op(B.transpose, dense_r, asserted_type=Dense)


def test_transpose_diag(diag1):
    check_un_op(B.transpose, diag1, asserted_type=Diagonal)


def test_transpose_const(const_r):
    check_un_op(B.transpose, const_r, asserted_type=Constant)


def test_transpose_lr(lr_r):
    check_un_op(B.transpose, lr_r, asserted_type=LowRank)


def test_transpose_wb(wb1):
    check_un_op(B.transpose, wb1, asserted_type=Woodbury)


def test_transpose_kron(kron_r):
    check_un_op(B.transpose, kron_r, asserted_type=Kronecker)
