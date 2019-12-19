import lab as B

from matrix import Dense, Diagonal, Constant, LowRank, Woodbury
# noinspection PyUnresolvedReferences
from ..util import (
    allclose,
    check_un_op,

    dense1,
    diag1,
    const1,
    lr1,
    wb1
)


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
