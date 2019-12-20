import lab as B

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


def test_diag_zero(zero1):
    check_un_op(B.diag, zero1)


def test_diag_dense(dense1):
    check_un_op(B.diag, dense1)


def test_diag_diag(diag1):
    check_un_op(B.diag, diag1)


def test_diag_const(const1):
    check_un_op(B.diag, const1)


def test_diag_lr(lr1):
    check_un_op(B.diag, lr1)


def test_diag_wb(wb1):
    check_un_op(B.diag, wb1)


def test_diag_kron(kron1):
    check_un_op(B.diag, kron1)
