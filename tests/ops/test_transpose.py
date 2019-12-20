import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury, Kronecker
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense1,
    diag1,
    const1,
    lr1,
    wb1,
    kron1
)


def test_transpose_dense(dense1):
    check_un_op(B.transpose, dense1, asserted_type=Dense)


def test_transpose_diag(diag1):
    check_un_op(B.transpose, diag1, asserted_type=Diagonal)


def test_transpose_const(const1):
    check_un_op(B.transpose, const1, asserted_type=Constant)


def test_transpose_lr(lr1):
    check_un_op(B.transpose, lr1, asserted_type=LowRank)


def test_transpose_wb(wb1):
    check_un_op(B.transpose, wb1, asserted_type=Woodbury)


def test_transpose_kron(kron1):
    check_un_op(B.transpose, kron1, asserted_type=Kronecker)
