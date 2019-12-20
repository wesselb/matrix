import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury, Kronecker, Zero
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    zero1,
    dense1,
    diag1,
    const1,
    lr1,
    wb1,
    kron1
)


def test_negative_zero(zero1):
    check_un_op(B.negative, zero1, asserted_type=Zero)


def test_negative_dense(dense1):
    check_un_op(B.negative, dense1, asserted_type=Dense)


def test_negative_diag(diag1):
    check_un_op(B.negative, diag1, asserted_type=Diagonal)


def test_negative_const(const1):
    check_un_op(B.negative, const1, asserted_type=Constant)


def test_negative_lr(lr1):
    check_un_op(B.negative, lr1, asserted_type=LowRank)


def test_negative_wb(wb1):
    check_un_op(B.negative, wb1, asserted_type=Woodbury)


def test_negative_kron(kron1):
    check_un_op(B.negative, kron1, asserted_type=Kronecker)
